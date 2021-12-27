include ../Makefile.common
TARGETS=fp_hex.test print_fragment.test breakdown.test time_breakdown.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
