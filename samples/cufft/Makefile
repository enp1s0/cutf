include ../Makefile.common
NVCCFLAGS+=-lcufft -lcurand
TARGETS=cufft.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
