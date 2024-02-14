include ../Makefile.common
NVCCFLAGS+=-lnvidia-ml
TARGETS=nvml.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
