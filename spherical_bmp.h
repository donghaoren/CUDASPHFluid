#ifndef CUDAFLUID_SPHERICAL_BMP_H
#define CUDAFLUID_SPHERICAL_BMP_H

#include "common.h"

struct SphericalEnvironment {
    void loadBMP(const char* filename);
    void release();
    Vector4 getColor(const Vector& direction);
    int width, height;
    Vector4* pixels;
};

#endif
