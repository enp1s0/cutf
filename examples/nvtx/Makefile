include ../Makefile.common
NVCCFLAGS+=
TARGETS=nvtx.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
