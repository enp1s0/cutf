NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_86 -I../../include
TARGETS=fp_hex.test print_fragment.test breakdown.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
