NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_75 -I../../include -Xcompiler="-Wall"
TARGETS=abs.test maxmin.test horizontal.test imath.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
