NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../ -lcusolver
TARGETS=geqrf.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
