NVCC=nvcc
NVCCFLAGS=-std=c++14 -I../../include -lcusolver -lcublas -lcurand
TARGETS=gesvd.test geqrf.test rsvd.test

NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
