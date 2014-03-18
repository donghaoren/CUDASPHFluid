#include <cstdio>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <cstring>
#include <cstdlib>
using namespace std;

#include "simulator.h"
#include "videomaker.h"

int main(int argc, char* argv[]) {
    Simulator sim;
    char profile_filename[256];
    strcpy(profile_filename, "profile.txt");
    if(argc == 2)
        strcpy(profile_filename, argv[1]);

    printf("Start simulation using profile '%s'\n", profile_filename);

    sim.initialize(profile_filename);
    if(sim.config.find("load") != sim.config.end())
        sim.loadState(sim.config["load"].c_str());

    VideoMaker* video = VideoMaker_Create("imgs/video", sim.render.width, sim.render.height);
    for(int i = 0; i < sim.max_frame; i++) {
        Timing tm("Frame");
        sim.runSimulation();
        sim.renderFrame();
        tm.measure("run");
        video->addFrame(sim.render.colors, sim.render.width, sim.render.height);
        tm.measure("write");
        fflush(stdout);
        if(i % 100 == 0) {
            char filename[256];
            sprintf(filename, "imgs/state.%04d.bin", i);
            sim.saveState(filename);
        }
    }
    sim.saveState("imgs/state.bin");
    video->finish();
    return 0;
}
