include ../Makefile.common
NVCCFLAGS+=-lnvrtc -lcuda
TARGETS=nvrtc.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
