#ifndef RENDERING2_H
#define RENDERING2_H

#include "fluid.h"
#include "spherical_bmp.h"

struct Render {
    FluidSystem &fs;
    Vector eye, at, up;
    number_t view_angle;

    int width, height;

    number_t refraction_rate_air;
    number_t refraction_rate_water;

    SphericalEnvironment environment;

    Vector4 *colors, *gpu_colors;
    float *depths, *gpu_depths;

    bool support_depth_fusion;

    void setViewport();
    void setEnvironment(const SphericalEnvironment& environment_);

    void render();

    Vector bbox_real_min, bbox_real_max;

    Render(FluidSystem &fs_);
};

#endif
