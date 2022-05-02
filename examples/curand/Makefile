include ../Makefile.common
NVCCFLAGS+=-lcurand
TARGETS=curand.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
