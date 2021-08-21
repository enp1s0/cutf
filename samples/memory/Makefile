NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include -lnvrtc -lcuda
TARGETS=async_copy.test async_malloc.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
