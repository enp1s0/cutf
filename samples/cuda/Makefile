NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include
TARGETS=cos.test type.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
