#include "fluid.h"
#include "simulator.h"
#include <vector>
#include <string>
#include <map>

struct ParticleRange {
    float x_min, x_max; // [y_min, y_max)
    bool inside(const Vector& x) { return x.x >= x_min && x.x < x_max; }
    bool inside_ghost(const Vector& x, float radius) { return x.x >= x_min - radius && x.x <= x_max + radius; }
};

class MPISimulator {
public:
    FluidSystem fluid;
    Render render;

    double frame_dt, current_t;
    Vector bbox_max, bbox_min;

    int max_frame;

    std::vector<PlaneConstrain> planes;

    std::vector<ParticleRange> thread_configs;

    void mpi_send_particles(int rank_from, int rank_to, std::vector<Particle>& ps);
    void mpi_recv_particles(int rank_from, int rank_to, std::vector<Particle>& ps);

    // Synchronize particles for compute force.
    // Retrieve ghost particles.
    void mpi_ghost_particles(float margin);

    // Exchange particles with other threads.
    void mpi_exchange_particles();

    // Distribute particles to threads.
    void mpi_distribute_particles();

    // Collect all particles to the same node (rank 0).
    // For rendering.
    void mpi_collect_particles();


    int nX, nY, nZ;
    double view_theta;
    double view_scale;
    double view_altitude;

    double particle_mass, particle_spacing;

    Vector eye, at, up;

    int rank, size;
    MPISimulator(int rank = 0, int size = 1);

    void initialize(const char* profile_path);
    void addDroplet();

    void changeView();

    void runSimulation();
    void renderFrame();

    void dumpData(int index);

    std::string save_path;
    bool saveState(const char* filename = 0);
    bool loadState(const char* filename = 0);
    void saveParticles();

    void computeTotalForce(double t);

    std::map<std::string, std::string> config;
};
