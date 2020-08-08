all: cublas.dir  cuda.dir  curand.dir  cusolver.dir  device.dir  error.dir  event.dir  mantissa.dir  math.dir  memory.dir  nvml.dir  nvrtc.dir

%.dir:
	cd $(shell basename $@ .dir);make
