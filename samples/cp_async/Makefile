NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_80 -I../../include
TARGETS=cp_async.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
