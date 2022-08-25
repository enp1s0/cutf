include ../Makefile.common
TARGETS=abs.test maxmin.test horizontal.test imath.test is.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
