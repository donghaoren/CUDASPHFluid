#include "fluid.h"
#include <math.h>
#include <stdlib.h>

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

void print_vector(const Vector& v) {
    printf("%.20lf,%.20lf,%.20lf\n", v.x, v.y, v.z);
}

void FluidSystem::defaultParameters() {
    resting_density = 1000;
    radius = 0.01;
    k_pressure = 2;
    k_viscosity = 0.35;
    k_surface_tension = 10;
    threshold_surface_tension = 1e-10;
}

void FluidSystem::computeForce() {
    Kernels kernel(radius);
    for(int i = 0; i < particles.size(); i++) {
        particles[i].a = Vector(0, 0, 0);
    }
    for(int i = 0; i < particles.size(); i++) {
        particles[i].density = 0;
        for(int j = 0; j < particles.size(); j++) {
            Vector dx = particles[i].x - particles[j].x;
            if(dx.len2() > radius * radius) continue;
            particles[i].density += kernel.poly6(dx) * particle_mass;
        }
        particles[i].pressure = k_pressure * (particles[i].density - resting_density);
    }

    // Particle formulation of the Navier-Stokes Equations
    for(int i = 0; i < particles.size(); i++) {
        Vector n(0);
        float grad2color = 0;
        for(int j = 0; j < particles.size(); j++) {
            Vector dx = particles[i].x - particles[j].x;
            if(dx.len2() > radius * radius) continue;
            // Pressure.
            number_t pressure = (particles[i].pressure + particles[j].pressure) / 2;
            Vector fp = kernel.spiky(dx) * (-pressure * particle_mass / particles[j].density);
            // Viscosity.
            Vector dv = particles[j].v - particles[i].v;
            Vector fv = dv * (particle_mass / particles[j].density * kernel.lap(dx)) * k_viscosity;
            // Surface Tension.
            n += kernel.spiky(dx) * (particle_mass / particles[j].density);
            grad2color += particle_mass / particles[j].density * kernel.lap(dx);
            // Add up forces.
            particles[i].a += (fp + fv) / particles[i].density;
            //particles[j].a -= (fp + fv) / particles[j].density;
        }
        /*
        if(n.len() > threshold_surface_tension) {
            Vector ft = n * (-k_surface_tension * grad2color / n.len());

            particles[i].a += (ft) / particles[i].density;
        }
        */
    }
}
