NVCCFLAGS=-O3 -m64 -arch sm_20
NVCC=nvcc
MPICC=mpicc
MPICCFLAGS=-O3
EXTRAS=
LINK=mpicc
LDFLAGS=-L/opt/cuda/current/lib64 -lcudart -lcublas
ifeq ($(SGE_CLUSTER_NAME), Lonestar4)
LDFLAGS=-L/opt/apps/cuda/5.0/lib64 -lcudart -lcublas
endif
ifeq ($(shell uname), Darwin)
NVCC=nvcc
CXX=clang++
NVCCFLAGS=-O3 -m64 -arch sm_20 -Xcompiler -stdlib=libstdc++
MPICCFLAGS=-stdlib=libstdc++
LDFLAGS=-Xlinker -framework,OpenGL,-framework,GLUT `mpicxx --showme:link`
LINK=nvcc
endif

all: project mpi_project interface

rkv.o: rkv.cpp rkv.h
	$(MPICC) rkv.cpp -c -o rkv.o $(MPICCFLAGS)

videomaker.o: videomaker.cpp videomaker.h common.h bitmap_image.hpp
	$(MPICC) videomaker.cpp -c -o videomaker.o $(MPICCFLAGS)

spherical_bmp.o: spherical_bmp.cpp spherical_bmp.h common.h
	$(MPICC) spherical_bmp.cpp -c -o spherical_bmp.o $(MPICCFLAGS)

fluid.o: fluid.cpp fluid.h common.h
	$(MPICC) fluid.cpp -c -o fluid.o $(MPICCFLAGS)

fluidgpu.o: fluidgpu.cu fluid.h common.h
	$(NVCC) fluidgpu.cu -c -o fluidgpu.o $(NVCCFLAGS)

rendering2.o: rendering2.cu rendering2.h fluid.h common.h spherical_bmp.h
	$(NVCC) rendering2.cu -c -o rendering2.o $(NVCCFLAGS)

rendering_diffuse.o: rendering_diffuse.cu rendering2.h fluid.h common.h spherical_bmp.h
	$(NVCC) rendering_diffuse.cu -c -o rendering_diffuse.o $(NVCCFLAGS)

simulator.o: simulator.cpp simulator.h fluid.h common.h rendering2.h spherical_bmp.h rkv.h
	$(MPICC) simulator.cpp -c -o simulator.o $(MPICCFLAGS)

mpi_simulator.o: mpi_simulator.cpp mpi_simulator.h fluid.h common.h simulator.h rendering2.h spherical_bmp.h rkv.h
	$(MPICC) mpi_simulator.cpp -c -o mpi_simulator.o $(MPICCFLAGS)

project.o: project.cpp simulator.h fluid.h common.h rendering2.h spherical_bmp.h videomaker.h
	$(MPICC) project.cpp -c -o project.o $(MPICCFLAGS)

mpi_project.o: mpi_project.cpp mpi_simulator.h fluid.h common.h simulator.h rendering2.h spherical_bmp.h videomaker.h
	$(MPICC) mpi_project.cpp -c -o mpi_project.o $(MPICCFLAGS)

interface.o: interface.cpp mpi_simulator.h fluid.h common.h simulator.h rendering2.h spherical_bmp.h
	$(MPICC) interface.cpp -c -o interface.o $(MPICCFLAGS)

project: project.o rkv.o fluid.o fluidgpu.o rendering2.o  videomaker.o spherical_bmp.o simulator.o
	$(LINK) project.o rkv.o fluid.o fluidgpu.o rendering2.o  videomaker.o spherical_bmp.o simulator.o -o project $(LDFLAGS)

mpi_project: mpi_project.o rkv.o fluid.o fluidgpu.o rendering2.o  videomaker.o spherical_bmp.o simulator.o mpi_simulator.o
	$(LINK) mpi_project.o rkv.o fluid.o fluidgpu.o rendering2.o  videomaker.o spherical_bmp.o simulator.o mpi_simulator.o -o mpi_project $(LDFLAGS)

interface: interface.o rkv.o fluid.o fluidgpu.o rendering2.o videomaker.o spherical_bmp.o simulator.o mpi_simulator.o
	$(LINK) interface.o rkv.o fluid.o fluidgpu.o rendering2.o videomaker.o spherical_bmp.o simulator.o mpi_simulator.o -o interface $(LDFLAGS)

clean:
	rm -rf project mpi_project interface *.o
# DO NOT DELETE
