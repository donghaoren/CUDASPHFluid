#include <cstdio>
#include <cmath>
#include <vector>
#include <deque>
#include <string>
#include <cstdlib>
using namespace std;

#ifdef __APPLE__
// If compiling in a Mac
#include <GLUT/GLUT.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#else
// In Windows

#include <GL/glut.h>

#endif

#include "mpi_simulator.h"

MPISimulator sim;

bool render_raytracing = true;

// Display function.
void display() {
    Timing tm("Display");
    // Clear
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(render_raytracing) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glDisable(GL_LIGHTING);
        glDisable(GL_LIGHT0);
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(-1, -1);
        sim.renderFrame();
        glDrawPixels(sim.render.width, sim.render.height, GL_RGBA,GL_FLOAT, sim.render.colors);

    } else {
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        glEnable(GL_DEPTH_TEST);

        // Initialize transformation.

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(90, (double)sim.render.width / sim.render.height, 0.0001, 10);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        gluLookAt(sim.eye.x, sim.eye.y, sim.eye.z, sim.at.x, sim.at.y, sim.at.z, sim.up.x, sim.up.y, sim.up.z);

        float v = sim.fluid.particle_mass / sim.fluid.resting_density;
        float r = pow(v / (4.0/3.0*3.1415926), 1.0/3);

        for(int i = 0; i < sim.fluid.particles.size(); i++) {
            glPushMatrix();
            glTranslated(sim.fluid.particles[i].x.x, sim.fluid.particles[i].x.y, sim.fluid.particles[i].x.z);
            glutSolidSphere(r, 5, 5);
            glPopMatrix();
        }
    }

    glFlush();
    glutSwapBuffers();
    tm.measure("total");
}


// Resize function.

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glutPostRedisplay();
}

void reshape() {
    int vp[4];
    glGetIntegerv(GL_VIEWPORT, vp);
    reshape(vp[2], vp[3]);
}

bool simulation_paused = true;
// Keyboard.
void keyboard(unsigned char key, int x, int y) {
    if(key == 'p') simulation_paused = !simulation_paused;
    if(key == 'q') {
        exit(0);
    }
    if(key == 'z') sim.view_theta += 0.1;
    if(key == 'x') sim.view_theta -= 0.1;
    if(key == 'a') sim.view_scale *= 1.05;
    if(key == 's') sim.view_scale /= 1.05;
    if(key == 'c') sim.view_altitude += 0.1;
    if(key == 'v') sim.view_altitude -= 0.1;
    if(key == 'w') sim.saveState();
    if(key == 'r') sim.loadState();
    sim.changeView();
    if(key == ' ') render_raytracing = !render_raytracing;
    if(key == '.') {
        sim.addDroplet();
    }
    reshape();
    glutPostRedisplay();
}

void timerfunc(int v) {
    if(!simulation_paused) {
        sim.runSimulation();
        sim.runSimulation();
        sim.runSimulation();
        glutPostRedisplay();
    }
	glutTimerFunc(40, timerfunc, 0);
}

int main(int argc, char* argv[]) {
    sim.initialize("profile.txt");

    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitWindowSize(sim.render.width, sim.render.height);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("CS280 Project Donghao Ren, Yanbo Ma, Di Ma");

    glClearColor(1, 1, 1, 1);
    glEnable(GL_BLEND);

    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Callback functions.
	glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(40, timerfunc, 0);
    // Main loop.
    glutMainLoop();
    return 0;
}
