include ../Makefile.common
TARGETS=min_exponent.test fp_mask.test reinterpret.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
