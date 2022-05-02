include ../Makefile.common
NVCCFLAGS+=-lcutensor
TARGETS=cutensor.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
