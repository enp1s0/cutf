include ../Makefile.common
NVCCFLAGS+=-lnvrtc -lcuda
TARGETS=error_test.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
