include ../Makefile.common
NVCCFLAGS+=-lcublas
TARGETS=gemm.test c8i_gemm.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
