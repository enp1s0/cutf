NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_86 -I../../include
TARGETS=graph.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
