NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../ -lcurand
TARGETS=curand.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
