#include <cutf/cutensor.hpp>
#include <cutf/error.hpp>

int main() {
	cutensorHandle_t handle;
	CUTF_CHECK_ERROR(cutensorInit(&handle));
}
