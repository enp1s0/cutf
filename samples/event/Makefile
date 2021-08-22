NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../ 
TARGETS=elapsed_time.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
