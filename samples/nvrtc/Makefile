NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include -lnvrtc -lcuda
TARGETS=nvrtc.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
