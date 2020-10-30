DIRS=$(shell . -maxdepth 1 -mindepth 1 -type d)

all: cublas.dir cuda.dir curand.dir cusolver.dir device.dir error.dir mantissa.dir math.dir memory.dir nvml.dir nvrtc.dir

%.dir:
	cd $(shell basename $@ .dir);make
