#include "spherical_bmp.h"
#include <stdio.h>
#include <math.h>

#define PIXEL_AT(x, y) pixels[(y) * width + (x)]
#define PIXEL_ATs(x, y) PIXEL_AT((x) % width, (y) % height)
#define PI 3.14159265358979323846264338

void SphericalEnvironment::loadBMP(const char* filename) {
    FILE* fin = fopen(filename, "rb");
    unsigned char* header = new unsigned char[100 * 1048576];
    int len = fread(header, 1, 100 * 1048576, fin);
    int offset = *(int*)(header + 10);
    width = *(int*)(header + 18);
    height = *(int*)(header + 22);
    if(height < 0) height = -height;
    unsigned char* array = header + offset;
    pixels = new Vector4[width * height];
    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++) {
            PIXEL_AT(x, y).z = (float)array[(y * width + x) * 3 + 0] / 255.0f;
            PIXEL_AT(x, y).y = (float)array[(y * width + x) * 3 + 1] / 255.0f;
            PIXEL_AT(x, y).x = (float)array[(y * width + x) * 3 + 2] / 255.0f;
            PIXEL_AT(x, y).w = 1.0f;
        }
    delete [] header;
}

void SphericalEnvironment::release() {
    delete [] pixels;
}

Vector4 SphericalEnvironment::getColor(const Vector& direction_) {
    Vector direction = direction_.normalize();
    Vector proj = direction;
    proj.z = 0;
    float theta = atan2(-direction.y, -direction.x);
    float phi = -atan(direction.z / proj.len());
    float xp = (theta / PI + 1.0) / 2.0 * width;
    float yp = (phi / (PI / 2.0) + 1.0) / 2.0 * height;
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
