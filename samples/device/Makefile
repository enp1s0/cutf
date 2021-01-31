NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_61 -I../../include
TARGETS=device_info.test specify_device.test

all: $(TARGETS)

%.test: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
