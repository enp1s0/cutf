include ../Makefile.common
NVCCFLAGS+=-lcublasLt
TARGETS=gemm.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
