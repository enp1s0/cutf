NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include
TARGETS=tf32.test fp64_mantissa_cutter_matmul.test fp32_mantissa_cutter_matmul.test comparison_to_half.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
