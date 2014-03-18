#ifndef CUDAFLUID_COMMON_H
#define CUDAFLUID_COMMON_H

#include <cmath>

typedef float number_t;

#ifdef __device__
#define __general__ inline __device__ __host__
#else
#define __general__ inline
#endif

struct Vector {
    number_t x, y, z;
    __general__ Vector() { }
    __general__ Vector(number_t v) { x = y = z = v; }
    __general__ Vector(number_t x_, number_t y_, number_t z_) : x(x_), y(y_), z(z_) { }
    __general__ Vector& operator += (const Vector& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __general__ Vector& operator -= (const Vector& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    __general__ Vector& operator *= (number_t s) { x *= s; y *= s; z *= s; return *this; }
    __general__ Vector& operator /= (number_t s) { x /= s; y /= s; z /= s; return *this; }
    __general__ Vector operator + (const Vector& v) const { return Vector(x + v.x, y + v.y, z + v.z); }
    __general__ Vector operator - (const Vector& v) const { return Vector(x - v.x, y - v.y, z - v.z); }
    __general__ Vector operator - () const { return Vector(-x, -y, -z); }
    __general__ Vector operator * (number_t s) const { return Vector(x * s, y * s, z * s); }
    __general__ Vector operator / (number_t s) const { return Vector(x / s, y / s, z / s); }
    __general__ number_t operator * (const Vector& v) const { return x * v.x + y * v.y + z * v.z; }
    __general__ number_t len2() const { return x * x + y * y + z * z; }
    __general__ number_t len() const { return std::sqrt(x * x + y * y + z * z); }
    __general__ Vector normalize() const { return *this / len(); }
    __general__ bool operator <= (const Vector& v) const { return x <= v.x && y <= v.y && z <= v.z; }
    __general__ bool operator >= (const Vector& v) const { return x >= v.x && y >= v.y && z >= v.z; }
    __general__ Vector cross(const Vector& v) const {
        return Vector(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
};

__general__ number_t num_min(number_t a, number_t b) { return a < b ? a : b; }
__general__ number_t num_max(number_t a, number_t b) { return a > b ? a : b; }
__general__ number_t clamp01(number_t a) { return a < 0 ? 0 : (a > 1 ? 1 : a); }

struct Vector4 {
    number_t x,y,z,w;//r,g,b,a
    __general__ Vector4() { };
    __general__ Vector4(Vector a,number_t b) { x = a.x; y = a.y; z = a.z; w = b; }
    __general__ Vector4(number_t v) { x = y = z = w = v; }
    __general__ Vector4(number_t x_, number_t y_, number_t z_, number_t w_) : x(x_), y(y_), z(z_) , w(w_) { }
    __general__ Vector4 operator + (const Vector4& v) const { return Vector4(x + v.x, y + v.y, z + v.z, w + v.w); }
    __general__ Vector4 operator - (const Vector4& v) const { return Vector4(x - v.x, y - v.y, z - v.z, w - v.w); }
    __general__ Vector xyz(){ return Vector(x, y, z); }
    __general__ Vector4 operator * (number_t s) const { return Vector4(x * s, y * s, z * s, w * s); }
};

#endif
