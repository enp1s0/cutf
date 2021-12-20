include ../Makefile.common
TARGETS=async_copy.test async_malloc.test malloc.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
