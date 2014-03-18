#include <cstdio>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <cstring>
#include <cstdlib>
using namespace std;

#include "mpi_simulator.h"
#include "videomaker.h"

#include "mpi.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char profile_filename[256];
    strcpy(profile_filename, "profile.txt");
    if(argc == 2)
        strcpy(profile_filename, argv[1]);

    printf("Start simulation using profile '%s'\n", profile_filename);

    printf("MPI: rank = %d, size = %d\n", rank, size);

    MPISimulator sim(rank, size);

    sim.initialize(profile_filename);

    VideoMaker* video;
    if(rank == 0) video = VideoMaker_Create("imgs/video", sim.render.width, sim.render.height);
    for(int i = 0; i < sim.max_frame; i++) {
        Timing tm("Frame");
        sim.runSimulation();
        sim.runSimulation();
        sim.runSimulation();
        tm.measure("run");

        sim.renderFrame();
        if(rank == 0) {
            video->addFrame(sim.render.colors, sim.render.width, sim.render.height);
            tm.measure("write");
        }
        if(i % 1000 == 0) {
            sim.dumpData(i);
        }
    }
    if(rank == 0) { video->finish(); delete video; }
    MPI_Finalize();
    return 0;
}
