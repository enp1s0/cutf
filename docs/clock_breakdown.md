# Clock level breakdown helper functions

## Usage
```cuda
__global__ void kernel(float* const ptr) {
	CUTF_CLOCK_BREAKDOWN_INIT(2);
	CUTF_CLOCK_BREAKDOWN_RECORD(0);

	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	const auto v = ptr[tid];
	ptr[tid] = v * v;

	CUTF_CLOCK_BREAKDOWN_RECORD(1);

	printf("%lld\n",
			CUTF_CLOCK_BREAKDOWN_DURATION(0, 1)
			);
}
```

### CUTF_CLOCK_BREAKDOWN_INIT(num_records)

This macro declares an array for timestamps.

### CUTF_CLOCK_BREAKDOWN_RECORD(index)

This macro sets clock on an element of the array.

### CUTF_CLOCK_BREAKDOWN_DURATION(start_index, end_index)

This macro calculates the duration between two timestamps given by array indices.
