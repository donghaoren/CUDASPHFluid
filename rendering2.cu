#include "rendering2.h"

#define MAX_THREAD_COUNT 512

Render::Render(FluidSystem &fs_) : fs(fs_) {
    view_angle = 90;
    colors = 0;
    gpu_colors = 0;
    refraction_rate_air = 1.0;
    refraction_rate_water = 1.5;
    support_depth_fusion = false;
}

void Render::setViewport() {
    printf("[Render] Viewport: %d %d\n", width, height);
    if(colors) {
        delete [] colors;
        cudaFree(gpu_colors);
    }
    colors = new Vector4[width * height];
    cudaMalloc((void**)&gpu_colors, sizeof(Vector4) * width * height);
}

#define PIXEL_AT(x, y) environment.pixels[(y) * environment.width + (x)]
#define PIXEL_ATs(x, y) PIXEL_AT((x) % environment.width, (y) % environment.height)
#define PI 3.14159265358979323846264338

struct RenderContext {
    int width, height;
    int volume_nx, volume_ny, volume_nz;    // volume size
    Vector bbox_min;  // bounding box.
    number_t voxel_size;

    Vector eye, at, up;
    number_t view_angle;

    number_t* gpu_volume;
    Vector4* gpu_colors;

    number_t surface_threshold;

    number_t refraction_rate_air, refraction_rate_water;

    SphericalEnvironment environment;

    __device__ Vector4 get_environment(Vector direction) {
        Vector proj = direction;
        proj.z = 0;
        float theta = atan2(-direction.y, -direction.x);
        float phi = -atan(direction.z / proj.len());
        float xp = (theta / PI + 1.0) / 2.0 * environment.width;
        float yp = (phi / (PI / 2.0) + 1.0) / 2.0 * environment.height;
        int x = xp;
        float sx = xp - x;
        int y = yp;
        float sy = yp - y;
        Vector4 p00 = PIXEL_ATs(x, y);
        Vector4 p01 = PIXEL_ATs(x, y + 1);
        Vector4 p11 = PIXEL_ATs(x + 1, y + 1);
        Vector4 p10 = PIXEL_ATs(x + 1, y);
        Vector4 p0 = p00 * (1.0 - sy) + p01 * sy;
        Vector4 p1 = p10 * (1.0 - sy) + p11 * sy;
        return p0 * (1.0 - sx) + p1 * sx;
    }

    inline __device__ number_t vol_at(int x, int y, int z) {
        if(x < 0 || y < 0 || z < 0 || x >= volume_nx || y >= volume_ny || z >= volume_nz) return 0;
        return gpu_volume[(z * volume_nx * volume_ny + y * volume_nx + x)];
    }

    inline __device__ number_t get_density(const Vector& p) {
        register float fx = (p.x - bbox_min.x) / voxel_size;
        register float fy = (p.y - bbox_min.y) / voxel_size;
        register float fz = (p.z - bbox_min.z) / voxel_size;
        register int nx = floor(fx);
        register int ny = floor(fy);
        register int nz = floor(fz);
        // get 8 corners.
        register number_t v000 = vol_at(nx    , ny    , nz    );
        register number_t v100 = vol_at(nx + 1, ny    , nz    );
        register number_t v010 = vol_at(nx    , ny + 1, nz    );
        register number_t v110 = vol_at(nx + 1, ny + 1, nz    );
        register number_t v001 = vol_at(nx    , ny    , nz + 1);
        register number_t v101 = vol_at(nx + 1, ny    , nz + 1);
        register number_t v011 = vol_at(nx    , ny + 1, nz + 1);
        register number_t v111 = vol_at(nx + 1, ny + 1, nz + 1);
        // get interpolation coefficient
        register number_t x0 = fx - nx;
        register number_t y0 = fy - ny;
        register number_t z0 = fz - nz;
        // interpolate
        register number_t v00 = x0 * v100 + (1 - x0) * v000;
        register number_t v01 = x0 * v101 + (1 - x0) * v001;
        register number_t v10 = x0 * v110 + (1 - x0) * v010;
        register number_t v11 = x0 * v111 + (1 - x0) * v011;
        // second
        register number_t v0 = y0 * v10 + (1 - y0) * v00;
        register number_t v1 = y0 * v11 + (1 - y0) * v01;
        // third
        return z0 * v1 + (1 - z0) * v0;
    }
    inline __device__ Vector get_normal(const Vector& p) {
        register float dxyz = -voxel_size / 2.0f;
        register float vACC = get_density(p + Vector(+dxyz, 0, 0));
        register float vBCC = get_density(p + Vector(-dxyz, 0, 0));
        register float vCAC = get_density(p + Vector(0, +dxyz, 0));
        register float vCBC = get_density(p + Vector(0, -dxyz, 0));
        register float vCCA = get_density(p + Vector(0, 0, +dxyz));
        register float vCCB = get_density(p + Vector(0, 0, -dxyz));
        return Vector(vACC - vBCC, vCAC - vCBC, vCCA - vCCB).normalize();
    }
    // Advance the ray so that it lays on the boundary of the bbox.
    // return -1 if no interaction and the ray is outside.
    inline __device__ number_t intersect_bbox(const Vector& p, const Vector& direction) {
        Vector bmin = bbox_min - Vector(voxel_size, voxel_size, voxel_size);
        Vector bmax = bbox_min + Vector(volume_nx, volume_ny, volume_nz) * voxel_size;
        // if inside, no intersection.
        if(p <= bmax && p >= bmin) return 0;
        // Test for x.
        if(p.x < bmin.x && direction.x > 0) {
            number_t t = (bmin.x - p.x) / direction.x;
            Vector r = p + direction * t;
            if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
                // there's only one possible, if we find it, return.
                return t;
            }
        }
        if(p.x > bmax.x && direction.x < 0) {
            number_t t = (bmax.x - p.x) / direction.x;
            Vector r = p + direction * t;
            if(r.y >= bmin.y && r.y <= bmax.y && r.z >= bmin.z && r.z <= bmax.z) {
                return t;
            }
        }
        if(p.y < bmin.y && direction.y > 0) {
            number_t t = (bmin.y - p.y) / direction.y;
            Vector r = p + direction * t;
            if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
                return t;
            }
        }
        if(p.y > bmax.y && direction.y < 0) {
            number_t t = (bmax.y - p.y) / direction.y;
            Vector r = p + direction * t;
            if(r.x >= bmin.x && r.x <= bmax.x && r.z >= bmin.z && r.z <= bmax.z) {
                return t;
            }
        }
        if(p.z < bmin.z && direction.z > 0) {
            number_t t = (bmin.z - p.z) / direction.z;
            Vector r = p + direction * t;
            if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
                return t;
            }
        }
        if(p.z > bmax.z && direction.z < 0) {
            number_t t = (bmax.z - p.z) / direction.z;
            Vector r = p + direction * t;
            if(r.x >= bmin.x && r.x <= bmax.x && r.y >= bmin.y && r.y <= bmax.y) {
                return t;
            }
        }
        // no intersection.
        return -1;
    }

    inline __device__ bool find_zero_crossing(const Vector& p, const Vector& direction, number_t& t_result, Vector& normal) {
        number_t t = intersect_bbox(p, direction);
        Vector bmin = bbox_min - Vector(voxel_size, voxel_size, voxel_size);
        Vector bmax = bbox_min + Vector(volume_nx, volume_ny, volume_nz) * voxel_size;
        if(t < 0) return false;
        t = t + 1e-5;
        number_t val_prev = get_density(p + direction * t) - surface_threshold;
        number_t step_size = voxel_size / 2.0f;
        for(;;) {
            Vector pos = p + direction * t;
            if(pos <= bmax && pos >= bmin) {
                // we are still in the volume.
                number_t val = get_density(pos) - surface_threshold;
                if(val * val_prev <= 0) {
                    // do a binary search.
                    if(val != 0) {
                        number_t tmin = t - step_size;
                        number_t tmax = t;
                        for(int index = 0; index < 15; index++) {
                            t = (tmin + tmax) / 2.0f;
                            number_t val_mid = get_density(p + direction * t) - surface_threshold;
                            if(fabs(val_mid) < 0.01f) {
                                break;
                            } else {
                                if(val * val_mid >= 0) {
                                    tmax = t;
                                    val = val_mid;
                                } else {
                                    tmin = t;
                                }
                            }
                        }
                        t_result = t;
                        normal = get_normal(p + direction * t);
                        return true;
                    }
                }
                val_prev = val;
            } else break;
            t += step_size;
        }
        return false;
    }
};

template<int depth> __device__ inline Vector4 ray_tracing(RenderContext& ctx, const Vector& p, const Vector& direction, number_t refraction_rate) {
    number_t t;
    Vector normal;
    if(ctx.find_zero_crossing(p, direction, t, normal)) {
        Vector newp = p + direction * t;
        number_t c1 = direction * normal;
        number_t next_rate;
        // Make sure c1 > 0, d and normal in opposite direction.
        if(c1 > 0) {
            next_rate = ctx.refraction_rate_air;
            normal = -normal;
        } else {
            next_rate = ctx.refraction_rate_water;
            c1 = -c1;
        }
        Vector reflected_direction = direction - normal * (direction * normal * 2);
        Vector4 color_reflection = ray_tracing<depth + 1>(ctx, newp, reflected_direction, refraction_rate);
        number_t n = refraction_rate / next_rate;
        number_t c2 = 1.0 - n * n * (1.0 - c1 * c1);
        if(c2 > 0) {
            c2 = sqrt(c2);
            Vector refracted_direction = (direction * n) + normal * (n * c1 - c2);
            Vector4 color_refraction = ray_tracing<depth + 1>(ctx, newp, refracted_direction, next_rate);
            number_t r1 = (refraction_rate * c1 - next_rate * c2) / (refraction_rate * c1 + next_rate * c2);
            number_t r2 = (next_rate * c1 - refraction_rate * c2) / (next_rate * c1 + refraction_rate * c2);
            number_t ratio = (r1 * r1 + r2 * r2) / 2.0f;
            return color_reflection * ratio + color_refraction * (1.0f - ratio);
        } else {
            // Total internal reflect.
            return color_reflection;
        }
    } else {
        return ctx.get_environment(direction);
    }
}

template<> __device__ inline Vector4 ray_tracing<2>(RenderContext& ctx, const Vector& p, const Vector& direction, number_t refraction_rate) {
    return ctx.get_environment(direction);
}

__global__ void compute_render(RenderContext ctx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % ctx.width;
    int y = idx / ctx.width;
    Vector4 color(0, 0, 0, 1);
    if(x >= ctx.width || y >= ctx.height) return;

    Vector center_direction = (ctx.at - ctx.eye).normalize();
    Vector ex = center_direction.cross(ctx.up).normalize();
    Vector ey = ex.cross(center_direction);
    number_t scale = tan(ctx.view_angle / 2.0f / 180.0f * PI) * 2;
    float tx = ((float)x + 0.5f - (float)ctx.width / 2.0f) / (float)ctx.width;
    float ty = ((float)y + 0.5f - (float)ctx.height / 2.0f) / (float)ctx.width;
    Vector direction = (center_direction + ex * scale * tx + ey * scale * ty).normalize();
    // Initial light.
    Vector p = ctx.eye;
    Vector normal;
    number_t refraction_rate = ctx.refraction_rate_air;
    if(ctx.get_density(p) > ctx.surface_threshold) refraction_rate = ctx.refraction_rate_water;

    ctx.gpu_colors[idx] = ray_tracing<0>(ctx, p, direction, refraction_rate);
}

#define gpuErrchkCritical(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=false)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void Render::setEnvironment(const SphericalEnvironment& env) {
    environment.width = env.width;
    environment.height = env.height;
    printf("[Render] Initialize environment: %d %d\n", env.width, env.height);
    cudaMalloc((void**)&environment.pixels, sizeof(Vector4) * env.width * env.height);
    gpuErrchkCritical(cudaGetLastError());
    cudaMemcpy(environment.pixels, env.pixels, sizeof(Vector4) * env.width * env.height, cudaMemcpyHostToDevice);
    gpuErrchkCritical(cudaGetLastError());
}

void Render::render() {
    RenderContext ctx;
    ctx.width = width;
    ctx.height = height;
    ctx.volume_nx = fs.density_nx;
    ctx.volume_ny = fs.density_ny;
    ctx.volume_nz = fs.density_nz;
    ctx.voxel_size = fs.voxel_size;
    ctx.bbox_min = fs.bbox_min;
    ctx.gpu_colors = gpu_colors;
    ctx.gpu_volume = fs.density_volume_cuda;
    ctx.refraction_rate_air = refraction_rate_air;
    ctx.refraction_rate_water = refraction_rate_water;
    ctx.eye = eye;
    ctx.at = at;
    ctx.up = up;
    ctx.view_angle = view_angle;

    ctx.environment = environment;

    ctx.surface_threshold = fs.resting_density / 10.0f;

    int total_threads = width * height;
    int block_count = total_threads / MAX_THREAD_COUNT;

    compute_render<<<block_count + 1, MAX_THREAD_COUNT>>>(ctx);
    cudaMemcpy(colors, gpu_colors, sizeof(Vector4) * width * height, cudaMemcpyDeviceToHost);
    gpuErrchkCritical(cudaGetLastError());
}
