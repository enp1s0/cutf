include ../Makefile.common
TARGETS=prefetch.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
