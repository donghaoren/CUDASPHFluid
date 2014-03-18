#include "fluid.h"

#include <algorithm>
#include <vector>
#include <cstdarg>

#define SAFE_Malloc(pointer, size) cudaMalloc((void**)&(pointer), size); gpuErrchkCritical(cudaGetLastError());

struct HashMap {
    std::vector<int> data;
    std::vector<hash_t> keys;
    std::vector<bool> used;
    int size;
    HashMap(int count) : data(count, 0), keys(count), used(count, false), size(count) {
    }
    inline int hash_function(hash_t key) {
        int r = (key * 2654435761) % size;
        if(r < 0) r += size;
        return r;
    }
    void insert(hash_t key, int val) {
        int pos = hash_function(key);
        while(used[pos]) {
            pos++;
            if(pos >= size) pos = 0;
        }
        data[pos] = val;
        keys[pos] = key;
        used[pos] = true;
    }
    int find(hash_t key) {
        int pos = hash_function(key);
        while(used[pos] && keys[pos] != key) {
            pos++;
            if(pos >= size) pos = 0;
        }
        if(used[pos]) return data[pos];
        return -1;
    }
};

#define MAX_BLOCK_THREADS 128
#define MAX_THREAD_COUNT 512
#define RATIO_GRID_RADIUS 3

struct Kernels {
    number_t d;
    number_t k_poly6, d2;
    number_t k_spiky;
    Kernels(number_t d_) {
        d = d_;
        d2 = d * d;
        number_t d6 = d2 * d2 * d2;
        number_t d9 = d6 * d2 * d;
        k_poly6 = 315.0 / 64.0 / 3.14159265358979323846264 / d9;
        k_spiky = 45.0 / 3.14159265358979323846264 / d6;
    }
    number_t poly6(Vector r) {
        number_t x = d2 - r.len2();
        return k_poly6 * (x * x * x);
    }
    Vector spiky(Vector r) {
        number_t l = sqrt(r.len2());
        if(l == 0) return Vector(0);
        return r * (-k_spiky * (d - l) * (d - l) / l);
    }
    number_t lap(Vector r) {
        return k_spiky * (d - sqrt(r.len2()));
    }
};

__global__ void compute_initialize(ParticleBuffers buf) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= buf.N) return;
    buf.a[i] = Vector(0, 0, 0);
}

#define hash_minxyz -0.2

inline __device__ __host__ hash_t hash_position(Vector pos, number_t radius) {
    number_t gridsize = radius * RATIO_GRID_RADIUS;
    hash_t ix = (pos.x - hash_minxyz) / gridsize;
    hash_t iy = (pos.y - hash_minxyz) / gridsize;
    hash_t iz = (pos.z - hash_minxyz) / gridsize;
    return (ix << 40) | (iy << 20) | (iz);
}

inline __device__ __host__ int3 hash_get_ints(hash_t hash) {
    int3 r;
    r.x = (hash >> 40) & 1048575;
    r.y = (hash >> 20) & 1048575;
    r.z = hash & 1048575;
    return r;
}

inline __device__ __host__ Vector hash_get_position(int3 i, number_t radius) {
    number_t gridsize = radius * RATIO_GRID_RADIUS;
    return Vector(
        i.x * gridsize + hash_minxyz,
        i.y * gridsize + hash_minxyz,
        i.z * gridsize + hash_minxyz
    );
}

inline __device__ __host__ bool hash_inside_block(Vector pos, hash_t hash, number_t radius) {
    number_t gridsize = radius * RATIO_GRID_RADIUS;
    number_t lbx = (number_t)((hash >> 40) & 1048575) * gridsize + hash_minxyz;
    number_t lby = (number_t)((hash >> 20) & 1048575) * gridsize + hash_minxyz;
    number_t lbz = (number_t)(hash & 1048575) * gridsize + hash_minxyz;
    lbx -= radius; lby -= radius; lbz -= radius;
    bool j1 = pos.x >= lbx && pos.y >= lby && pos.z >= lbz;
    lbx += gridsize * 2;
    lby += gridsize * 2;
    lbz += gridsize * 2;
    return j1 && pos.x <= lbx && pos.y <= lby && pos.z <= lbz;
}

__global__ void compute_hash_table(ParticleBuffers buf) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= buf.N) return;
    buf.hash_table[i].hash = hash_position(buf.x[i], buf.radius);
    buf.hash_table[i].index = i;
}

__global__ void compute_sort_particles(ParticleBuffers buf) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= buf.N) return;
    int pi = buf.hash_table[i].index;
    buf.sorted_info[i].x = buf.x[pi];
    buf.sorted_info[i].v = buf.v[pi];
}

__global__ void compute_unsort_particles(ParticleBuffers buf) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= buf.N) return;
    int pi = buf.hash_table[i].index;
    buf.pressure[pi] = buf.sorted_info[i].pressure;
    buf.density[pi] = buf.sorted_info[i].density;
}

__global__ void compute_pressure(ParticleBuffers buf, int start) {
    register number_t density = 0;

    __shared__ Vector neighbor_x[MAX_BLOCK_THREADS];

    register BlockInfo block = buf.block_table[blockIdx.x + start];
    register int i = threadIdx.x;
    register float radius2 = buf.radius2;
    int ptr_i = block.start + i;
    register Vector xi;
    if(i < block.count)
        xi = buf.sorted_info[ptr_i].x;
    for(int c = block.neighbor_start; c < block.neighbor_end; c++) {
        BlockInfo neighbor = buf.block_table[buf.block_neighbors[c]];
        // Copy neighbors into shared memory.
        if(i < neighbor.count)
            neighbor_x[i] = buf.sorted_info[neighbor.start + i].x;
        __syncthreads();
        // Sync and do the work.
        if(i < block.count) {
            // Compute density for each neighbor.
            for(int j = 0; j < neighbor.count; j++) {
                register Vector dx = xi - neighbor_x[j];
                register number_t l2 = radius2 - dx.len2();
                if(l2 > 0) {
                    density += l2 * l2 * l2;
                }
            }
        }
        __syncthreads();
    }
    density *= buf.k_poly6_mass;
    if(i < block.count) {
        buf.sorted_info[ptr_i].density = density;
        buf.sorted_info[ptr_i].pressure = buf.k_pressure * (density - buf.resting_density);
        buf.sorted_info[ptr_i].spiky_m_over_density = buf.k_spiky * buf.particle_mass / density;
    }
}

__global__ void compute_force(ParticleBuffers buf, int start) {
    register Vector f = 0;
    //register Vector normal = 0;

    __shared__ ParticleInfo neighbor_shared[MAX_BLOCK_THREADS];

    register BlockInfo block = buf.block_table[blockIdx.x + start];
    register int i = threadIdx.x;
    register int ptr_i = block.start + i;
    register ParticleInfo pi;
    register float k_viscosity = buf.k_viscosity;
    register float radius2 = buf.radius2;

    //register Vector grad_color(0);
    //register float lap_color = 0;

    if(i < block.count)
        pi = buf.sorted_info[ptr_i];

    for(int c = block.neighbor_start; c < block.neighbor_end; c++) {
        register BlockInfo neighbor = buf.block_table[buf.block_neighbors[c]];
        if(i < neighbor.count) {
            neighbor_shared[i] = buf.sorted_info[neighbor.start + i];
        }
        __syncthreads();
        // Sync and do the work.
        if(i < block.count) {
            // Compute density for each neighbor.
            for(int j = 0; j < neighbor.count; j++) {
                register ParticleInfo pj = neighbor_shared[j];
                register Vector dx = pi.x - pj.x;
                register number_t l2 = dx.len2();
                if(l2 < radius2 && l2 > 0) {
                    // Pressure.
                    register number_t l = sqrtf(l2);
                    register number_t radius_minus_l = buf.radius - l;
                    register number_t pressure = (pi.pressure + pj.pressure) / 2;
                    register Vector fp = dx * (radius_minus_l / l * pressure * pj.spiky_m_over_density);
                    // Viscosity.
                    register Vector dv = pj.v - pi.v;
                    register Vector fv = dv * (pj.spiky_m_over_density * k_viscosity);
                    //normal += fp / pressure * radius_minus_l;

                    //grad_color += dx * (pj.spiky_m_over_density * (radius_minus_l * radius_minus_l / l));
                    //lap_color += pj.spiky_m_over_density * radius_minus_l;

                    // Add up forces.
                    f += (fp + fv) * (radius_minus_l);
                }
            }
        }
        __syncthreads();
    }
    if(i < block.count) {
        //grad_color *= buf.radius;
        //buf.sorted_info[ptr_i].normal = grad_color * buf.radius;
        buf.a[buf.hash_table[ptr_i].index] = f / pi.density;
    }
}
/*
__global__ void compute_surface_tension(ParticleBuffers buf) {
    register Vector f = 0;
    //register Vector normal = 0;

    __shared__ Vector normals[MAX_BLOCK_THREADS];

    register BlockInfo block = buf.block_table[blockIdx.x];
    register int i = threadIdx.x;
    register int ptr_i = block.start + i;
    register Vector ni;

    if(i < block.count)
        ni = buf.sorted_info[ptr_i].normal;

    for(int c = block.neighbor_start; c < block.neighbor_end; c++) {
        register BlockInfo neighbor = buf.block_table[buf.block_neighbors[c]];
        if(i < neighbor.count) {
            normals[i] = buf.sorted_info[neighbor.start + i].normal;
        }
        __syncthreads();
        // Sync and do the work.
        if(i < block.count) {
            // Compute density for each neighbor.
            for(int j = 0; j < neighbor.count; j++) {
                register Vector nj = normals[j];
                f += ni - nj;
            }
        }
        __syncthreads();
    }
    if(i < block.count) {
        f = f * -0.1 * buf.particle_mass;
        buf.a[buf.hash_table[ptr_i].index] += f / buf.sorted_info[ptr_i].density;
    }
}
*/

__global__ void compute_density_volume_zero(number_t* density_volume, int N, int nx, int ny, int nz, int start) {
    int i = threadIdx.x + (blockIdx.x + start) * blockDim.x;
    if(i < N) {
        density_volume[i] = 0;
    }
}

__global__ void compute_density_volume(
    int N, Vector* positions,
    number_t* density_volume,
    int nx, int ny, int nz, float voxel_size, Vector bbox_min, float radius, float k_poly6_mass
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= N) return;
    Vector xi = positions[i];
    int x0 = (xi.x - bbox_min.x) / voxel_size;
    int y0 = (xi.y - bbox_min.y) / voxel_size;
    int z0 = (xi.z - bbox_min.z) / voxel_size;
    int r = radius / voxel_size + 2;
    float radius2 = radius * radius;
    for(int shx = -r; shx <= r; shx++)
    for(int shy = -r; shy <= r; shy++)
    for(int shz = -r; shz <= r; shz++) {
        int px = x0 + shx;
        int py = y0 + shy;
        int pz = z0 + shz;
        if(px >= 0 && py >= 0 && pz >= 0 && px < nx && py < ny && pz < nz) {
            Vector xj = Vector(px, py, pz) * voxel_size + bbox_min;
            Vector dx = xi - xj;
            number_t l2 = radius2 - dx.len2();
            if(l2 > 0) {
                float density = l2 * l2 * l2;
                density *= k_poly6_mass;
                atomicAdd(density_volume + pz * nx * ny + py * nx + px, density);
            }
        }
    }
}

#ifndef _WIN32
#include <sys/time.h>
double getPreciseTime() {
    timeval t;
    gettimeofday(&t, 0);
    double s = t.tv_sec;
    s += t.tv_usec / 1000000.0;
    return s;
}
#else
#include <windows.h>
double getPreciseTime() {
	LARGE_INTEGER data, frequency;
    QueryPerformanceCounter(&data);
    QueryPerformanceFrequency(&frequency);
    return (double)data.QuadPart / (double)frequency.QuadPart;
	//return 0;
}
#endif

int _timing_depth = 0;
Timing::Timing(const std::string _name) {
    name = _name;
    t = getPreciseTime();
    _timing_depth += 1;
}
Timing::~Timing() { _timing_depth -= 1; }

void Timing::print(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);
    for(int i = 0; i < _timing_depth - 1; i++) printf(" ");
    printf("[%s] %s\n", name.c_str(), buf);
    fflush(stdout);
}

void Timing::measure(const char* s) {
    double dt = getPreciseTime() - t;
    for(int i = 0; i < _timing_depth - 1; i++) printf(" ");
    printf("[%s] %s: %lf ms\n", name.c_str(), s, dt * 1000);
    fflush(stdout);
    t = getPreciseTime();
}

#define gpuErrchkCritical(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      fflush(stderr);
      if (abort) exit(code);
   }
}

void FluidSystem::computeForceGPU() {
    Timing tm("ComputeForce");
    // Initialize parameters.
    int N = particles.size();
    tm.print("N = %d", N);

    if(particles.size() == 0) return;

    ensureMemory();


    Kernels kernel(radius);
    cuda_buffers.k_viscosity = k_viscosity;
    cuda_buffers.resting_density = resting_density;
    cuda_buffers.radius = radius;
    cuda_buffers.k_pressure = k_pressure;
    cuda_buffers.k_surface_tension = k_surface_tension;
    cuda_buffers.threshold_surface_tension = threshold_surface_tension;
    cuda_buffers.particle_mass = particle_mass;
    cuda_buffers.radius2 = radius * radius;
    cuda_buffers.k_poly6_mass = kernel.k_poly6 * particle_mass;
    cuda_buffers.k_spiky = kernel.k_spiky;
    cuda_buffers.N = N;
    cpu_buffers.N = N;

    // Copy particles into cpu buffer.
    for(int i = 0; i < N; i++) {
        cpu_buffers.x[i] = particles[i].x;
        cpu_buffers.v[i] = particles[i].v;
    }
    // Copy particles' x and v to gpu buffer.
    cudaMemcpy(cuda_buffers.x, cpu_buffers.x, sizeof(Vector) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_buffers.v, cpu_buffers.v, sizeof(Vector) * N, cudaMemcpyHostToDevice);
    gpuErrchkCritical(cudaGetLastError());

    // Block and thread count for division.
    int blockCount = N / MAX_THREAD_COUNT;
    int threadCount = MAX_THREAD_COUNT;
    if(threadCount * blockCount < N) blockCount += 1;

    tm.measure("Start");
    tm.print("Block count: %d, thread count: %d", blockCount, threadCount);

    compute_initialize<<< blockCount, threadCount >>>(cuda_buffers);
    gpuErrchkCritical(cudaGetLastError());
    compute_hash_table<<< blockCount, threadCount >>>(cuda_buffers);
    gpuErrchkCritical(cudaGetLastError());

    cudaThreadSynchronize();
    tm.measure("Init & Hash");

    cudaMemcpy(cpu_buffers.hash_table, cuda_buffers.hash_table, sizeof(HashTag) * N, cudaMemcpyDeviceToHost);
    gpuErrchkCritical(cudaGetLastError());

    std::sort(cpu_buffers.hash_table, cpu_buffers.hash_table + N);

    tm.measure("CPU Sort");

    N_blocks = 0;
    max_block_size = MAX_BLOCK_THREADS;
    int N_neighbor_list;
    {
        int i = 0;
        while(i < N) {
            int j = i;
            while(j < N && cpu_buffers.hash_table[i].hash == cpu_buffers.hash_table[j].hash) {
                j++;
            }
            int count = j - i;
            for(int k = 0; k < count; k++) {
                cpu_buffers.hash_infos[i + k].count = count;
            }
            i = j;
        }
        int block_idx = 0;
        for(int i = 0; i < N; i += cpu_buffers.hash_infos[i].count) {
            int c = cpu_buffers.hash_infos[i].count;
            int n_subblocks = c / max_block_size;
            if(c % max_block_size > 0) n_subblocks += 1;
            for(int j = 0; j < n_subblocks; j++) {
                cpu_buffers.block_table[block_idx].start = i + j * max_block_size;
                cpu_buffers.block_table[block_idx].count = max_block_size;
                cpu_buffers.block_table[block_idx].hash = cpu_buffers.hash_table[i].hash;
                if(j == n_subblocks - 1) cpu_buffers.block_table[block_idx].count = c - j * max_block_size;
                block_idx += 1;
            }
        }
        //tm.measure("Generate Blocks - A");
        N_blocks = block_idx;
        int nlist_ptr = 0;
        int real_max_block_size = 0;
        HashMap block_map(N_blocks * 2);
        for(int i = 0; i < N_blocks; i++) {
            block_map.insert(cpu_buffers.block_table[i].hash, i);
        }
        //tm.measure("Generate Blocks - B");
        for(int i = 0; i < N_blocks; i++) {
            hash_t hash0 = cpu_buffers.block_table[i].hash;
            // Find neighbors.
            cpu_buffers.block_table[i].neighbor_start = nlist_ptr;
            for(int ta = -1; ta <= 1; ta++) for(int tb = -1; tb <= 1; tb++) for(int tc = -1; tc <= 1; tc++) {
                hash_t hash = hash0 + ta * ((hash_t)1 << 40) + tb * ((hash_t)1 << 20) + tc;
                int idx = block_map.find(hash); //hashtag_binary_search(cpu_buffers.block_table, N_blocks, hash);
                if(idx >= 0) {
                    while(idx > 0 && cpu_buffers.block_table[idx - 1].hash == hash) idx--;
                    while(idx < N_blocks && cpu_buffers.block_table[idx].hash == hash) {
                        cpu_buffers.block_neighbors[nlist_ptr++] = idx++;
                    }
                }
            }
            cpu_buffers.block_table[i].neighbor_end = nlist_ptr;
            if(real_max_block_size < cpu_buffers.block_table[i].count)
                real_max_block_size = cpu_buffers.block_table[i].count;
        }
        max_block_size = real_max_block_size;
        tm.print("Particle count: %d, Block count: %d, Max block size: %d", N, N_blocks, max_block_size);
        N_neighbor_list = nlist_ptr;
    }
    cudaMemcpy(cuda_buffers.hash_table, cpu_buffers.hash_table, sizeof(HashTag) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_buffers.hash_infos, cpu_buffers.hash_infos, sizeof(HashInfo) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_buffers.block_table, cpu_buffers.block_table, sizeof(BlockInfo) * N_blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_buffers.block_neighbors, cpu_buffers.block_neighbors, sizeof(int) * N_neighbor_list, cudaMemcpyHostToDevice);
    gpuErrchkCritical(cudaGetLastError());

    tm.measure("Generate Blocks");

    tm.print("GPU Config: blocks = %d, max_bs = %d, bc = %d, tc = %d", N_blocks, max_block_size, blockCount, threadCount);

    compute_sort_particles<<< blockCount, threadCount >>>(cuda_buffers);
    gpuErrchkCritical(cudaGetLastError());

    cudaThreadSynchronize();

    tm.measure("Reorder particles");

    int max_blocks = 16384;
    for(int start = 0; start < N_blocks; start += max_blocks) {
        int this_size = N_blocks - start;
        if(this_size > max_blocks) this_size = max_blocks;
        compute_pressure<<< this_size, max_block_size >>>(cuda_buffers, start);
        gpuErrchkCritical(cudaGetLastError());
    }

    cudaThreadSynchronize();
    tm.measure("Pressure");

    for(int start = 0; start < N_blocks; start += max_blocks) {
        int this_size = N_blocks - start;
        if(this_size > max_blocks) this_size = max_blocks;
        compute_force<<< this_size, max_block_size >>>(cuda_buffers, start);
        gpuErrchkCritical(cudaGetLastError());
    }

    cudaThreadSynchronize();
    tm.measure("Force");

    compute_unsort_particles<<< blockCount, threadCount >>>(cuda_buffers);
    gpuErrchkCritical(cudaGetLastError());

    cudaMemcpy(cpu_buffers.a, cuda_buffers.a, sizeof(Vector) * N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(cpu_buffers.normal, cuda_buffers.normal, sizeof(Vector) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_buffers.pressure, cuda_buffers.pressure, sizeof(number_t) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_buffers.density, cuda_buffers.density, sizeof(number_t) * N, cudaMemcpyDeviceToHost);
    gpuErrchkCritical(cudaGetLastError());

    for(int i = 0; i < N; i++) {
        particles[i].a = cpu_buffers.a[i];
        particles[i].pressure = cpu_buffers.pressure[i];
        particles[i].density = cpu_buffers.density[i];
    }

    cudaThreadSynchronize();
    gpuErrchkCritical(cudaGetLastError());
    tm.measure("Copy Back");
}

void FluidSystem::initVolumeParameters(const Vector& min, const Vector& max) {
    hash_t hash = hash_position(min, radius);
    bbox_min_hash = hash;
    bbox_min = hash_get_position(hash_get_ints(hash), radius);
    voxel_size = radius * 0.5;
    Vector bbox_size = (max - bbox_min) / voxel_size + Vector(1, 1, 1);
    density_nx = bbox_size.x;
    density_ny = bbox_size.y;
    density_nz = bbox_size.z;
    printf("Volume parameters: %f %dx%dx%d, total: %d voxels.\n", voxel_size, density_nx, density_ny, density_nz, density_nx * density_ny * density_nz);
    printf("Allocate volume: %ld bytes.\n", sizeof(number_t) * density_nx * density_ny * density_nz);
    SAFE_Malloc(density_volume_cuda, sizeof(number_t) * density_nx * density_ny * density_nz);
}

void FluidSystem::computeDensityVolumeGPU() {
    Timing tm("ComputeDensityVolume");

    int volume_size = density_nx * density_ny * density_nz;
    int blockCount = volume_size / MAX_THREAD_COUNT;
    if(volume_size % MAX_THREAD_COUNT != 0) blockCount += 1;

    int max_blocks = 16384;
    for(int start = 0; start < blockCount; start += max_blocks) {
        int this_size = blockCount - start;
        if(this_size > max_blocks) this_size = max_blocks;
        compute_density_volume_zero<<< this_size, MAX_THREAD_COUNT >>>(density_volume_cuda, volume_size, density_nx, density_ny, density_nz, start);
        gpuErrchkCritical(cudaGetLastError());
    }

    if(particles.size() > 0) {
        ensureMemory();
        int N = particles.size();
        // Copy particles into gpu buffer.
        for(int i = 0; i < N; i++) {
            cpu_buffers.x[i] = particles[i].x;
        }
        cudaMemcpy(cuda_buffers.x, cpu_buffers.x, sizeof(Vector) * N, cudaMemcpyHostToDevice);
        gpuErrchkCritical(cudaGetLastError());

        blockCount = particles.size() / MAX_THREAD_COUNT;
        if(particles.size() % MAX_THREAD_COUNT != 0) blockCount += 1;

        float _radius = radius * 2;
        Kernels rk(_radius);

        compute_density_volume<<< blockCount, MAX_THREAD_COUNT >>>(
            N, cuda_buffers.x, density_volume_cuda,
            density_nx, density_ny, density_nz,
            voxel_size, bbox_min, _radius, rk.k_poly6 * particle_mass);

        gpuErrchkCritical(cudaGetLastError());

    }

    //cudaMemcpy(density_volume, density_volume_cuda, sizeof(number_t) * volume_size, cudaMemcpyDeviceToHost);

    tm.measure("total");
}

void FluidSystem::ensureMemory() {
    if(particles.size() > allocated_size) {
        int N = particles.size() * 1.1;
        if(allocated_size != 0) {
            cudaFinalize();
        }
        cudaInitialize(N);
        allocated_size = N;
    }
}

FluidSystem::FluidSystem() {
    allocated_size = 0;
    density_volume_cuda = 0;
}

FluidSystem::~FluidSystem() {
    if(allocated_size > 0) cudaFinalize();
}

void FluidSystem::cudaInitialize(int N) {
    //cudaGetLastError();
    gpuErrchkCritical(cudaGetLastError());

    long long int total_memory = (
        sizeof(Vector) * 3 + sizeof(number_t) * 2 +
        sizeof(HashTag) + sizeof(HashInfo) +
        sizeof(BlockInfo) + sizeof(int) * 27 +
        sizeof(ParticleInfo)
    ) * N;
    Timing tm("CUDA Allocate");
    tm.print("Allocate Memory: %d, %lld bytes, %.2lf MB.", N, total_memory, (double)total_memory / 1048576.0);

	SAFE_Malloc(cuda_buffers.x, sizeof(Vector) * N);
    SAFE_Malloc(cuda_buffers.v, sizeof(Vector) * N);
    SAFE_Malloc(cuda_buffers.a, sizeof(Vector) * N);
    //SAFE_Malloc(cuda_buffers.normal, sizeof(Vector) * N);
    SAFE_Malloc(cuda_buffers.pressure, sizeof(number_t) * N);
    SAFE_Malloc(cuda_buffers.density, sizeof(number_t) * N);
    SAFE_Malloc(cuda_buffers.hash_table, sizeof(HashTag) * N);
    SAFE_Malloc(cuda_buffers.hash_infos, sizeof(HashInfo) * N);
    SAFE_Malloc(cuda_buffers.block_table, sizeof(BlockInfo) * N);
    SAFE_Malloc(cuda_buffers.block_neighbors, sizeof(int) * N * 27);
    SAFE_Malloc(cuda_buffers.sorted_info, sizeof(ParticleInfo) * N);
    cpu_buffers.x = new Vector[N];
    cpu_buffers.v = new Vector[N];
    cpu_buffers.a = new Vector[N];
    //cpu_buffers.normal = new Vector[N];
    cpu_buffers.pressure = new number_t[N];
    cpu_buffers.density = new number_t[N];
    cpu_buffers.hash_table = new HashTag[N];
    cpu_buffers.hash_infos = new HashInfo[N];
    cpu_buffers.block_table = new BlockInfo[N];
    cpu_buffers.block_neighbors = new int[N * 27];
    gpuErrchkCritical(cudaGetLastError());
}

void FluidSystem::cudaFinalize() {
    //cudaGetLastError();
    gpuErrchkCritical(cudaGetLastError());
    Timing tm("CUDA Free");
    tm.print("Free memory.");
    cudaFree(cuda_buffers.x);
    cudaFree(cuda_buffers.v);
    cudaFree(cuda_buffers.a);
    //cudaFree(cuda_buffers.normal);
    cudaFree(cuda_buffers.pressure);
    cudaFree(cuda_buffers.density);
    cudaFree(cuda_buffers.hash_table);
    cudaFree(cuda_buffers.hash_infos);
    cudaFree(cuda_buffers.block_table);
    cudaFree(cuda_buffers.block_neighbors);
    cudaFree(cuda_buffers.sorted_info);
    //cudaFree(density_volume_triangles_cuda);
    delete [] cpu_buffers.x;
    delete [] cpu_buffers.v;
    delete [] cpu_buffers.a;
    //delete [] cpu_buffers.normal;
    delete [] cpu_buffers.pressure;
    delete [] cpu_buffers.density;
    delete [] cpu_buffers.hash_table;
    delete [] cpu_buffers.hash_infos;
    delete [] cpu_buffers.block_table;
    delete [] cpu_buffers.block_neighbors;
    gpuErrchkCritical(cudaGetLastError());
    allocated_size = 0;
}
void FluidSystem::cudaExit() {
    cudaDeviceReset();
    gpuErrchkCritical(cudaGetLastError());
}

void FluidSystem::cudaSelectDevice(int rank) {
    int device_count;
    cudaGetDeviceCount(&device_count);
    int device_id = rank % device_count;
    cudaSetDevice(device_id);
    printf("[MPI Thread: %d] Set device: %d\n", rank, device_id);
    gpuErrchkCritical(cudaGetLastError());
}
