#include "fluid.h"
#include "rendering2.h"
#include <vector>
#include <string>
#include <map>

struct PlaneConstrain {
    Vector n, p;
    float v;
    PlaneConstrain() { v = 0; }
    void apply(Particle& particle) {
        float bound_eps = 1e-6;
        double distance = (particle.x - p) * (n);
        if(distance < bound_eps && particle.v * (n) >= 0 && particle.a * (n) < 0) {
            particle.a = particle.a - n * (particle.a * (n));
        }
    }
    void correctPosition(Particle& particle, double frame_dt) {
        float bound_eps = 1e-6;
        float k_restitution = 0.5;
        if((particle.x - p) * (n) < bound_eps && particle.v * (n) < v) {
            particle.v = particle.v - n * (n * (particle.v) * (1 + k_restitution) - v);
            particle.vh = particle.v + particle.a * (frame_dt / 2.0);
        }
    }
};

class Simulator {
public:
    FluidSystem fluid;
    Render render;

    double frame_dt, current_t;
    Vector bbox_max, bbox_min;

    int max_frame;

    std::vector<PlaneConstrain> planes;

    int nX, nY, nZ;
    double view_theta;
    double view_scale;
    double view_altitude;

    double particle_mass, particle_spacing;

    Vector eye, at, up;

    Simulator();

    void initialize(const char* profile_path);
    void addDroplet();

    void changeView();

    void runSimulation();
    void renderFrame();

    std::string save_path;
    bool saveState(const char* filename = 0);
    bool loadState(const char* filename = 0);
    void saveParticles();

    void computeTotalForce(double t);

    std::map<std::string, std::string> config;
};
