// Particle based fluid simulation.

#ifndef CUDAFLUID_H
#define CUDAFLUID_H

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>

#include "common.h"

typedef long long int hash_t;

struct Particle {
    Vector x, v, a;
    Vector vh;                  // vh for leapfrog integration.
    number_t pressure, density; // mass and density.
};

struct HashTag {
    hash_t hash;
    int index;
    bool operator < (const HashTag& h) const {
        return hash < h.hash;
    }
    bool operator == (const HashTag& h) const {
        return hash == h.hash;
    }
};

struct HashInfo {
    int count;
};

struct BlockInfo {
    hash_t hash;
    int start, count;
    int neighbor_start, neighbor_end;
};

struct Timing {
    double t;
    std::string name;
    Timing(const std::string _name);
    void measure(const char* s);
    void print(const char* fmt, ...);
    ~Timing();
};

struct ParticleInfo {
    Vector x, v;
    float pressure, spiky_m_over_density, density;
    //Vector normal;
};

struct ParticleBuffers {
    Vector *x, *v, *a; //, *normal;
    number_t *pressure, *density;

    ParticleInfo *sorted_info;

    number_t radius;  // radius of the kernel.
    number_t k_viscosity; // viscosity term.
    number_t k_pressure; // stiffness.
    number_t k_surface_tension, threshold_surface_tension; // kappa and normal threshold.
    number_t resting_density; // resting density.
    number_t particle_mass;
    number_t k_poly6_mass, k_spiky, radius2;

    HashTag* hash_table;
    HashInfo* hash_infos;
    BlockInfo* block_table;
    int* block_neighbors;

    int N;
};

struct Triangle {
    Vector a, b, c;
    bool valid;
};

struct FluidSystem {
    number_t radius;  // radius of the kernel.
    number_t k_viscosity; // viscosity term.
    number_t k_pressure; // stiffness.
    number_t k_surface_tension, threshold_surface_tension; // kappa and normal threshold.
    number_t resting_density; // resting density.
    number_t particle_mass;
    std::vector<Particle> particles;

    ParticleBuffers cuda_buffers;
    ParticleBuffers cpu_buffers;

    // Density volume.
    number_t* density_volume_cuda;
    Vector bbox_min;
    int bbox_min_hash;
    number_t voxel_size;
    int density_nx, density_ny, density_nz;
    Triangle* density_volume_triangles_cuda;
    Triangle* density_volume_triangles;

    FluidSystem();
    ~FluidSystem();

    int allocated_size;
    void cudaInitialize(int N);
    void cudaFinalize();
    void ensureMemory();

    void defaultParameters();
    void computeForce(); // slow cpu version.
    void computeForceGPU();
    void initVolumeParameters(const Vector& bbox_min, const Vector& bbox_max);
    void computeDensityVolumeGPU();
    void computeDensityIsoSurface(number_t isolevel);
    void cudaExit();

    void cudaSelectDevice(int rank);

    int N_blocks;
    int max_block_size;
};

#endif
