#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <thread>

#include "Utils/Utils.h"
#include "Camera/Camera.h"
#include "Object/Sphere.h"
#include "Object/ObjectWorld.h"
#include "Material/Lambertian.h"
#include "Material/Dielectric.h"
#include "Material/Metal.h"
#include "config.h"
#include "common.cuh"

/* 全局参数设置 */
namespace {
	/* 图片设置 */
	const int Image_Width = 400;			// 图片宽度
	const int Image_Height = 225;			// 图片高度
	Color data[Image_Width][Image_Height];	// 图片数据
	const double aspect = 1.0f * Image_Width / Image_Height;	// 宽高比

	/* 世界设置 */
	Color background(0.5, 0.7, 1.0);// 背景颜色
	ObjectWorld world(background);	// 世界中的物体

	/* 相机设置 */
	Vec3 from(3, 3, 2);
	Vec3 at(0, 0, -1);
	double focus = (from - at).length();
	Camera main_camera(from, at, Vec3(0, 1, 0), 20, aspect, 2.0, focus);	// 主相机
}

void AddObjects() {
	world.Add(std::make_shared<Sphere>(
		Point3(0, 0, -1),
		0.5,
		std::make_shared<Lambertian>(Color(0.1, 0.2, 0.5)))
	);
	world.Add(std::make_shared<Sphere>(
		Point3(0, -100.5, -1),
		100,
		std::make_shared<Metal>(Color(0.8, 0.8, 0), 0.3))
	);
	world.Add(std::make_shared<Sphere>(
		Point3(-1, 0, -1),
		0.5,
		std::make_shared<Metal>(Color(0, 0, 1)))
	);
	world.Add(std::make_shared<Sphere>(
		Point3(1, 0, -1),
		0.5,
		std::make_shared<Lambertian>(Color(1, 0, 0)))
	);
}

__global__ void Render(Color data[], const ObjectWorld& world, const Camera& main_camera) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int ID = i * blockDim.x + j;

	// 每个像素随机采样 samples_per_pixel 次, 并取平均值
	for (int cnt = 0; cnt < samples_per_pixel; cnt++) {
		// 当前像素的[0,1]坐标
		auto u = double(i + Random::random_double_01()) / Image_Width;
		auto v = double(j + Random::random_double_01()) / Image_Height;

		// 获取当前像素对应的光线
		Ray r = main_camera.GetRay(u, v);

		// 计算光线得到的颜色
		data[ID] += world.GetColor(r);
	}
	data[ID] /= samples_per_pixel;

	// gamma 矫正
	data[ID].gamma();
}

int main(void) {
	// 1. 设置GPU设备
	SetGPU();

	// 2. 添加物体
	AddObjects();

	Color* dev_data = nullptr;
	ObjectWorld* dev_world = nullptr;
	Camera* dev_camera = nullptr;
	ErrorCheck(cudaMalloc(&dev_data, sizeof(data)), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc(&dev_world, sizeof(world)), __FILE__, __LINE__);
	ErrorCheck(cudaMalloc(&dev_camera, sizeof(main_camera)), __FILE__, __LINE__);


	// 2. 调用核函数在设备中进行计算
	cudaEvent_t start, stop;
	ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
	ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
		
	ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
	cudaEventQuery(start);

	Render<<<Image_Width, Image_Height >>>(dev_data, *dev_world, *dev_camera);

	ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
	ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);

	float elapsed_time;
	ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);
	printf("Time elapsed: %f ms\n", elapsed_time);

	ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
	ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);

	// 3. 计算结果从设备中拷贝到主机
	ErrorCheck(cudaMemcpy(data, dev_data, sizeof(data), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	// 3. 验证计算结果
	for (int i = 0; i < 2; i++)
		printf("%.2f %.2f %.2f\n", data[i][0][0], data[i][0][1], data[i][0][2]);

	// 4. 重置GPU设备
	cudaFree(dev_data);
	cudaFree(dev_world);
	cudaFree(dev_camera);
	cudaDeviceReset();
	return 0;
}