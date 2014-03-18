#include "mpi_simulator.h"
#include "rkv.h"
#include <mpi.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <stdlib.h>
using namespace std;

MPISimulator::MPISimulator(int rank_, int size_) : render(fluid) {
    rank = rank_;
    size = size_;
    fluid.cudaSelectDevice(rank);
}

void MPISimulator::initialize(const char* profile_path) {
    // Here we read parameters from the profile file.
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

    // Initialize all particles, in zero-th node.
    if(rank == 0) {
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
    }

    // Initialize bounding box.
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

    thread_configs.clear();
    float xmax_prev = -1e10;
    for(int i = 0; i < size; i++) {
        ParticleRange r;
        r.x_min = xmax_prev;
        float sc = (float)(i + 1) / (float)size;
        r.x_max = bbox_max.x * sc + bbox_min.x * (1 - sc);
        xmax_prev = r.x_max;
        thread_configs.push_back(r);
    }
    thread_configs[thread_configs.size() - 1].x_max = 1e10;

    Vector render_bbox_min = bbox_min;
    Vector render_bbox_max = bbox_max;
    render_bbox_min.x = fmax(bbox_min.x, thread_configs[rank].x_min - 1e-5);
    render_bbox_max.x = fmin(bbox_max.x, thread_configs[rank].x_max + 1e-5);


    printf("Particle mass: %lf, spacing: %lf\n", particle_mass, particle_spacing);
    printf("Bounding box: %f %f %f - %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);
    printf("Radius: %f\n", fluid.radius);

    // Initialize volume for root thread only.
    if(render.support_depth_fusion) {
        fluid.initVolumeParameters(render_bbox_min - Vector(fluid.radius * 2), render_bbox_max + Vector(fluid.radius * 2));
        render.bbox_real_min = render_bbox_min;
        render.bbox_real_max = render_bbox_max;
    } else {
        if(rank == 0) {
            fluid.initVolumeParameters(bbox_min, bbox_max);
        }
    }

    mpi_distribute_particles();

    // Setup internal structures.
    fluid.computeForceGPU();

    // Load environmental map.
    SphericalEnvironment env;
    env.loadBMP(bg_filename.c_str());
    render.setEnvironment(env);
    env.release();

    // Setup viewport.
    up = Vector(0, 0, 1);
    at = (bbox_max + bbox_min) / 2.0;
    at.z = particle_spacing * nZ / 2;
    eye = at + Vector(cos(view_theta), sin(view_theta), view_altitude) * view_scale;
    render.eye = eye;
    render.at = at;
    render.up = up;
    render.setViewport();
}

void MPISimulator::changeView() {
    up = Vector(0, 0, 1);
    at = (bbox_max + bbox_min) / 2.0;
    at.z = particle_spacing * nZ / 2;
    eye = at + Vector(cos(view_theta), sin(view_theta), view_altitude) * view_scale;
    render.eye = eye;
    render.at = at;
    render.up = up;
}

void MPISimulator::addDroplet() {
}

void MPISimulator::computeTotalForce(double t) {
    FluidSystem& tf = fluid;
    tf.computeForceGPU();
    vector<Particle>& ps = tf.particles;
    for(int i = 0; i < ps.size(); i++) {
        ps[i].a += Vector(0, 0, -9.8);
        planes[1].p.y = (1.0 - cos(t * 10)) / 20;
        planes[1].v = 10 * sin(t * 10) / 20;
        for(int k = 0; k < planes.size(); k++) {
            planes[k].apply(ps[i]);
        }
    }
}

// Perform one frame of simulation.
void MPISimulator::runSimulation() {
    Timing tm("SimulationStep");

    // TODO: Sync with other processors: send particles within the ghost region,
    // append them to tf.particles.
    int original_count = fluid.particles.size();
    mpi_ghost_particles(fluid.radius * 2);

    // Step 1: Compute forces for all particles.
    computeTotalForce(current_t);

    // TODO: Strip extra particles before updating positions..
    fluid.particles.resize(original_count);

    // Step 2: Update particle position with leapfrog integration.
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

    // TODO: move particles, make sure each thread has its own set of particles.
    // probably move thread boundaries.
    mpi_exchange_particles();

    current_t += frame_dt;
    tm.measure("total");
}

void MPISimulator::renderFrame() {
    Timing tm("Render");
    if(render.support_depth_fusion) {
        int original_count = fluid.particles.size();
        mpi_ghost_particles(fluid.radius * 2);
        fluid.computeDensityVolumeGPU();
        render.render();
        fluid.particles.resize(original_count);
        if(rank == 0) {
            Vector4 *other_colors = new Vector4[render.width * render.height];
            number_t *other_depths = new number_t[render.width * render.height];
            for(int i = 1; i < size; i++) {
                MPI_Status s;
                MPI_Recv(other_colors, sizeof(Vector4) * render.width * render.height, MPI_BYTE, i, 0, MPI_COMM_WORLD, &s);
                MPI_Recv(other_depths, sizeof(number_t) * render.width * render.height, MPI_BYTE, i, 0, MPI_COMM_WORLD, &s);
                for(int i = 0; i < render.width * render.height; i++) {
                    if(render.depths[i] > other_depths[i]) render.colors[i] = other_colors[i];
                }
            }
            delete [] other_colors;
            delete [] other_depths;
        } else {
            MPI_Send(render.colors, sizeof(Vector4) * render.width * render.height, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(render.depths, sizeof(number_t) * render.width * render.height, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        mpi_collect_particles();
        if(rank == 0) {
            fluid.computeDensityVolumeGPU();
            render.render();
        }
        mpi_distribute_particles();
    }
    tm.measure("total");
}

bool MPISimulator::saveState(const char* filename) {
    std::string fn = filename ? filename : save_path + "/state.bin";
    FILE* fout = fopen(fn.c_str(), "wb");
    if(fout) {
        fwrite(&fluid.particles[0], sizeof(Particle), fluid.particles.size(), fout);
        fclose(fout);
        return true;
    }
    return false;
}

void MPISimulator::dumpData(int index) {
    char fn[256];
    sprintf(fn, "dump-%06d.%02d.data", index, rank);
    FILE* fout = fopen(fn, "wb");
    if(fout) {
        int size = fluid.particles.size();
        fwrite(&size, sizeof(int), 1, fout);
        fwrite(&fluid.particles[0], sizeof(Particle), size, fout);
        fclose(fout);
    }
}

bool MPISimulator::loadState(const char* filename) {
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


void MPISimulator::mpi_send_particles(int rank_from, int rank_to, std::vector<Particle>& ps) {
    if(rank_from != rank) return;
    int count = ps.size();
    MPI_Send(&count, 1, MPI_INT, rank_to, 0, MPI_COMM_WORLD);
    MPI_Send(&ps[0], ps.size() * sizeof(Particle), MPI_BYTE, rank_to, 0, MPI_COMM_WORLD);
}

void MPISimulator::mpi_recv_particles(int rank_from, int rank_to, std::vector<Particle>& ps) {
    if(rank_to != rank) return;
    MPI_Status s;
    int count;
    MPI_Recv(&count, 1, MPI_INT, rank_from, 0, MPI_COMM_WORLD, &s);
    ps.resize(count);
    MPI_Recv(&ps[0], ps.size() * sizeof(Particle), MPI_BYTE, rank_from, 0, MPI_COMM_WORLD, &s);
}

void MPISimulator::mpi_distribute_particles() {
    Timing tm("MPI");
    if(rank == 0) {
        tm.print("[MPI] Distribute particles: %d to %d threads.", fluid.particles.size(), thread_configs.size());
        vector<vector<Particle> > packets(thread_configs.size());
        for(int i = 0; i < fluid.particles.size(); i++) {
            Particle& p = fluid.particles[i];
            for(int k = 0; k < thread_configs.size(); k++) {
                if(thread_configs[k].inside(p.x)) {
                    packets[k].push_back(p);
                }
            }
        }
        fluid.particles = packets[0];
        for(int i = 1; i < thread_configs.size(); i++) {
            mpi_send_particles(0, i, packets[i]);
        }
    } else {
        mpi_recv_particles(0, rank, fluid.particles);
    }
    tm.measure("finish");
}

void MPISimulator::mpi_collect_particles() {
    Timing tm("MPI");
    if(rank == 0) {
        vector<Particle> ps;
        for(int i = 1; i < thread_configs.size(); i++) {
            mpi_recv_particles(i, 0, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
        tm.print("[MPI] Collect particles: %d from %d threads.", fluid.particles.size(), thread_configs.size());
    } else {
        mpi_send_particles(rank, 0, fluid.particles);
        fluid.particles.clear();
    }
    tm.measure("finish");
}

void MPISimulator::mpi_exchange_particles() {
    Timing tm("MPI_Exchange");
    // Collect everything we need to send.
    int my_particle_count = fluid.particles.size();
    vector<vector<Particle> > packets(thread_configs.size());
    for(int i = 0; i < my_particle_count; i++) {
        Particle& p = fluid.particles[i];
        for(int k = 0; k < thread_configs.size(); k++) {
            if(thread_configs[k].inside(p.x)) {
                packets[k].push_back(p);
            }
        }
    }
    fluid.particles = packets[rank];

    int total_send = 0;
    for(int k = 0; k < thread_configs.size(); k++) {
        if(k != rank)
            total_send += packets[k].size();
    }
    tm.print("%d - send: %d", rank, total_send);
    // Send to nearby threads.
    // TODO: for long jumps, particle will be lost...
    if(rank % 2 == 0) {
        if(rank + 1 < size) {
            vector<Particle> ps;
            mpi_send_particles(rank, rank + 1, packets[rank + 1]);
            mpi_recv_particles(rank + 1, rank, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
        if(rank - 1 >= 0) {
            vector<Particle> ps;
            mpi_recv_particles(rank - 1, rank, ps);
            mpi_send_particles(rank, rank - 1, packets[rank - 1]);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
    } else {
        if(rank - 1 >= 0) {
            vector<Particle> ps;
            mpi_recv_particles(rank - 1, rank, ps);
            mpi_send_particles(rank, rank - 1, packets[rank - 1]);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
        if(rank + 1 < size) {
            vector<Particle> ps;
            mpi_send_particles(rank, rank + 1, packets[rank + 1]);
            mpi_recv_particles(rank + 1, rank, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
    }
    tm.measure("finish");
}

void MPISimulator::mpi_ghost_particles(float margin) {
    Timing tm("MPI_Ghost");
    // Collect everything we need to send.
    int my_particle_count = fluid.particles.size();
    vector<vector<Particle> > packets(thread_configs.size());
    for(int i = 0; i < my_particle_count; i++) {
        Particle& p = fluid.particles[i];
        for(int k = 0; k < thread_configs.size(); k++) {
            if(k == rank) continue; // don't send to it self.
            if(thread_configs[k].inside_ghost(p.x, margin)) {
                packets[k].push_back(p);
            }
        }
    }
    int total_send = 0;
    for(int k = 0; k < thread_configs.size(); k++) {
        total_send += packets[k].size();
    }
    tm.print("%d - send: %d", rank, total_send);
    // Send to nearby threads.
    if(rank % 2 == 0) {
        if(rank + 1 < size) {
            vector<Particle> ps;
            mpi_send_particles(rank, rank + 1, packets[rank + 1]);
            mpi_recv_particles(rank + 1, rank, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
        if(rank - 1 >= 0) {
            vector<Particle> ps;
            mpi_recv_particles(rank - 1, rank, ps);
            mpi_send_particles(rank, rank - 1, packets[rank - 1]);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
    } else {
        if(rank - 1 >= 0) {
            vector<Particle> ps;
            mpi_recv_particles(rank - 1, rank, ps);
            mpi_send_particles(rank, rank - 1, packets[rank - 1]);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
        if(rank + 1 < size) {
            vector<Particle> ps;
            mpi_send_particles(rank, rank + 1, packets[rank + 1]);
            mpi_recv_particles(rank + 1, rank, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
    }
    tm.measure("finish");

    /*
    int my_particle_count = fluid.particles.size();
    for(int send_thread = 0; send_thread < thread_configs.size(); send_thread++) {
        if(rank == send_thread) {
            int total_send = 0;
            vector<vector<Particle> > packets(thread_configs.size());
            for(int i = 0; i < my_particle_count; i++) {
                Particle& p = fluid.particles[i];
                for(int k = 0; k < thread_configs.size(); k++) {
                    if(k == send_thread) continue;
                    if(thread_configs[k].inside_ghost(p.x, fluid.radius)) {
                        packets[k].push_back(p);
                    }
                }
            }
            for(int k = 0; k < thread_configs.size(); k++) {
                if(k == send_thread) continue;
                total_send += packets[k].size();
                mpi_send_particles(send_thread, k, packets[k]);
            }
            tm.print("[MPI] Ghost: Send: %d rank = %d", total_send, rank);
        } else {
            vector<Particle> ps;
            mpi_recv_particles(send_thread, rank, ps);
            fluid.particles.insert(fluid.particles.end(), ps.begin(), ps.end());
        }
    }*/
}
