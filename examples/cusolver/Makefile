include ../Makefile.common
NVCCFLAGS+=-lcusolver -lcublas -Xcompiler=-fopenmp
TARGETS=gesvd.test potrf.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
