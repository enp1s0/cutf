# Smart pointers

## CUDA Device/Host Memory

- `cudaMalloc` / `cudaFree`
```cpp
cutf::cuda::memory::get_device_unique_ptr<type>(size)
```

- `cudaMallocHost` / `cudaFreeHost`
```cpp
cutf::cuda::memory::get_host_unique_ptr<type>(size)
```

## CUDA Stream

- `cudaCreateStream` / `cudaDestroyStream`
```cpp
cutf::cuda::stream::get_stream_unique_ptr(device_id = 0)
```

## cuBLAS Handle

- `cublasCreate` / `cublasDestroy`
```cpp
cutf::cublas::get_cublas_unique_ptr(device_id = 0)
```
