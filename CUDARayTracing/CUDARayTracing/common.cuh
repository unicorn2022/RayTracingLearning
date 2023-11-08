#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <algorithm>

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber) {
	if (error_code != cudaSuccess) {
		printf("CUDA error:\n");
		printf("code = %d, name = %s, description = %s\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code));
		printf("file = %s, line = %d\n", filename, lineNumber);
	}
	return error_code;
}

void SetGPU() {
	// 检测设备数量
	int deviceCount = 0;
	auto error = ErrorCheck(cudaGetDeviceCount(&deviceCount), __FILE__, __LINE__);

	if (error != cudaSuccess || deviceCount == 0)
		printf("[ERROR] No GPU!\n"), exit(-1);
	else
		printf("The count of GPUs is %d\n", deviceCount);

	// 设置执行设备
	int dev = 0;
	error = cudaSetDevice(dev);
	if (error != cudaSuccess)
		printf("[ERROR] Set device failed!\n"), exit(-1);
	else
		printf("Set GPU %d for computing\n", dev);
}