include ../Makefile.common
TARGETS=prefetch.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -g

clean:
	rm -f $(TARGETS)
