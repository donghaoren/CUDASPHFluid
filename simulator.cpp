#include "simulator.h"
#include "rkv.h"
#include <stdio.h>
#include <string>
#include <math.h>
#include <stdlib.h>
using namespace std;

Simulator::Simulator() : render(fluid) {
}

void Simulator::initialize(const char* profile_path) {
    fluid.defaultParameters();
    FILE* f = fopen(profile_path, "rt");
    char _pname[256];

    max_frame = 1000;

    nX = nY = nZ = 16;
    view_theta = 0;
    view_scale = 0.3;
    view_altitude = 0.4;

    render.width = 400;
    render.height = 300;

    frame_dt = 1.0 / 1000;
    current_t = 0;

    double sim_volume = 1.0;

    string bg_filename = "bg.bmp";
    save_path = "imgs/";

    while(fscanf(f, "%s", _pname) != EOF) {
        string pname = _pname;
        if(pname == "particles")
            fscanf(f, "%d %d %d", &nX, &nY, &nZ);
        else if(pname == "view-theta")
            fscanf(f, "%lf", &view_theta);
        else if(pname == "view-altitude")
            fscanf(f, "%lf", &view_altitude);
        else if(pname == "view-scale")
            fscanf(f, "%lf", &view_scale);
        else if(pname == "volume")
            fscanf(f, "%lf", &sim_volume);
        else if(pname == "k-pressure")
            fscanf(f, "%f", &fluid.k_pressure);
        else if(pname == "k-viscosity")
            fscanf(f, "%f", &fluid.k_viscosity);
        else if(pname == "k-surface-tension")
            fscanf(f, "%f", &fluid.k_surface_tension);
        else if(pname == "threshold-surface-tension")
            fscanf(f, "%f", &fluid.threshold_surface_tension);
        else if(pname == "width")
            fscanf(f, "%d", &render.width);
        else if(pname == "height")
            fscanf(f, "%d", &render.height);
        else if(pname == "frames")
            fscanf(f, "%d", &max_frame);
        else if(pname == "dt")
            fscanf(f, "%lf", &frame_dt);
        else if(pname == "bg") {
            char fn[256];
            fscanf(f, "%s", fn);
            bg_filename = fn;
        } else if(pname == "config") {
            char key[256];
            char val[256];
            fscanf(f, "%s %s", key, val);
            config[key] = val;
        } else if(pname == "save-path") {
            char fn[256];
            fscanf(f, "%s", fn);
            save_path = fn;
        } else {
            printf("Invalid parameter: %s\n", _pname);
            exit(-1);
        }
    }

    particle_mass = sim_volume / (nX * nY * nZ);
    particle_spacing = pow(particle_mass / fluid.resting_density, 1.0 / 3.0) * 0.9;

    fluid.particle_mass = particle_mass;
    fluid.radius = particle_spacing * 1.74;

    for(int i = 0; i < nX; i++) {
        for(int j = 0; j < nY; j++) {
            for(int k = 0; k < nZ; k++) {
                Particle p;
                double si = i * particle_spacing;
                double sj = j * particle_spacing;
                double sk = k * particle_spacing;
                p.x = Vector(si, sj, sk);
                p.v = Vector(0);
                fluid.particles.push_back(p);
            }
        }
    }

    {
        PlaneConstrain pl;
        pl.n = Vector(0, 0, 1);
        pl.p = Vector(0, 0, 0);
        planes.push_back(pl);
        pl.n = Vector(0, 1, 0);
        pl.p = Vector(0, 0, 0);
        planes.push_back(pl);
        pl.n = Vector(1, 0, 0);
        pl.p = Vector(0, 0, 0);
        planes.push_back(pl);
        pl.n = Vector(0, -1, 0);
        pl.p = Vector(0, nY * particle_spacing * 2, 0);
        planes.push_back(pl);
        pl.n = Vector(-1, 0, 0);
        pl.p = Vector(nX * particle_spacing, 0, 0);
        planes.push_back(pl);
    }

    bbox_min = Vector(0);
    bbox_max = Vector(nX * particle_spacing, nY * particle_spacing * 2, nZ * particle_spacing * 5);
    bbox_min -= Vector(particle_spacing * 10);
    bbox_max += Vector(particle_spacing * 10);

    printf("Particle mass: %lf, spacing: %lf\n", particle_mass, particle_spacing);
    printf("Bounding box: %f %f %f - %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);
    printf("Radius: %f\n", fluid.radius);

    fluid.initVolumeParameters(bbox_min, bbox_max);

    fluid.computeForceGPU();
    fluid.computeDensityVolumeGPU();

    SphericalEnvironment env;
    env.loadBMP(bg_filename.c_str());
    render.setEnvironment(env);
    env.release();

    up = Vector(0, 0, 1);
    at = (bbox_max + bbox_min) / 2.0;
    at.z = particle_spacing * nZ / 2;
    eye = at + Vector(cos(view_theta), sin(view_theta), view_altitude) * view_scale;
    render.eye = eye;
    render.at = at;
    render.up = up;
    render.setViewport();
}

void Simulator::changeView() {
    up = Vector(0, 0, 1);
    at = (bbox_max + bbox_min) / 2.0;
    at.z = particle_spacing * nZ / 2;
    eye = at + Vector(cos(view_theta), sin(view_theta), view_altitude) * view_scale;
    render.eye = eye;
    render.at = at;
    render.up = up;
}

void Simulator::addDroplet() {
}

void Simulator::computeTotalForce(double t) {
    FluidSystem& tf = fluid;

    tf.computeForceGPU();

    vector<Particle>& ps = tf.particles;
    for(int i = 0; i < ps.size(); i++) {
        ps[i].a += Vector(0, 0, -9.8);
        //planes[1].p.y = sin(t * 6) / 10;
        //planes[1].v = 8 * cos(t * 6) / 10;
        for(int k = 0; k < planes.size(); k++) {
            planes[k].apply(ps[i]);
        }
    }
}

// Perform one frame of simulation.
void Simulator::runSimulation() {
    Timing tm("RunStep");
    computeTotalForce(current_t);
    vector<Particle>& ps = fluid.particles;
    for(int i = 0; i < ps.size(); i++) {
        if(current_t == 0) {
            ps[i].vh = ps[i].v + ps[i].a * (frame_dt / 2.0);
            ps[i].v += ps[i].a * frame_dt;
            ps[i].x += ps[i].vh * frame_dt;
        } else {
            ps[i].vh += ps[i].a * frame_dt;
            ps[i].v = ps[i].vh + ps[i].a * (frame_dt / 2.0);
            ps[i].x += ps[i].vh * frame_dt;
        }
        for(int k = 0; k < planes.size(); k++) {
            planes[k].correctPosition(ps[i], frame_dt);
        }
    }
    current_t += frame_dt;
    tm.measure("total");
}

void Simulator::renderFrame() {
    Timing tm("Render");
    fluid.computeDensityVolumeGPU();
    render.render();
    tm.measure("total");
}

bool Simulator::saveState(const char* filename) {
    std::string fn = filename ? filename : save_path + "/state.bin";
    FILE* fout = fopen(fn.c_str(), "wb");
    if(fout) {
        fwrite(&fluid.particles[0], sizeof(Particle), fluid.particles.size(), fout);
        fclose(fout);
        return true;
    }
    return false;
}

bool Simulator::loadState(const char* filename) {
    std::string fn = filename ? filename : save_path + "/state.bin";
    FILE* fin = fopen(fn.c_str(), "rb");
    if(fin) {
        fread(&fluid.particles[0], sizeof(Particle), fluid.particles.size(), fin);
        fluid.computeForceGPU();
        fclose(fin);
        return true;
    }
    return false;
}

