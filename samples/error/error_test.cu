#include <iostream>
#include <cutf/cuda.hpp>
#include <cutf/error.hpp>

int main(){
	CUTF_CHECK_ERROR_M(cudaErrorInvalidValue, "Here :)");
}
