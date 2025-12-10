include ../Makefile.common
NVCCFLAGS+=-lcusolver -lcublas -Xcompiler=-fopenmp
TARGETS=gesvd.test potrf.test gels.test gesvdj.test geqrf.test xsyev_batched.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
