include ../Makefile.common
NVCCFLAGS+=-lcusolver -lcublas
TARGETS=gesvd.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
