#include <iostream>
#include <cutf/cuda.hpp>
#include <cutf/error.hpp>

int main(){
	CUTF_HANDLE_ERROR_M(cudaErrorInvalidValue, "Here :)");
}
