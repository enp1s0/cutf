NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include -lcublas
TARGETS=gemm.test c8i_gemm.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
