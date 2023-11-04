#pragma warning(disable:4996)
#include <iostream>
#include <iomanip>
#include "Utils/Utils.h"
#include "Camera/Camera.h"
#include "Object/Sphere.h"
#include "Object/ObjectWorld.h"
#include "Material/Lambertian.h"
#include "Material/Dielectric.h"
#include "Material/Metal.h"
#include "config.h"
#include <ctime>

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

void Render() {
	for (int j = 0; j < Image_Height; j++) {
		double completed = (j * 100) / Image_Height;
		std::cerr << "\r已完成: " << std::setw(3) << completed << "%" << std::flush;
		std::cerr << "\t剩余行数: " << std::setw(3) << Image_Height - j << std::flush;

		for (int i = 0; i < Image_Width; i++) {
			// 每个像素随机采样 samples_per_pixel 次, 并取平均值
			for (int cnt = 0; cnt < samples_per_pixel; cnt++) {
				// 当前像素的[0,1]坐标
				auto u = double(i + Random::random_double_01()) / Image_Width;
				auto v = double(j + Random::random_double_01()) / Image_Height;

				// 获取当前像素对应的光线
				Ray r = main_camera.GetRay(u, v);

				// 计算光线得到的颜色
				data[i][j] += world.GetColor(r);
			}
			data[i][j] /= samples_per_pixel;

			// gamma 矫正
			data[i][j].gamma();
		}
	}
}

int main() {
	freopen("../output.ppm", "w", stdout);

	// 渲染图片
	auto t = clock();
	Render();
	std::cerr << "\n图片渲染完成, 耗时 " << clock() - t << "ms\n";
	t = clock();

	// 输出图片
	std::cout << "P3\n" << Image_Width << " " << Image_Height << "\n255\n";
	for (int j = Image_Height - 1; j >= 0; j--)
		for (int i = 0; i < Image_Width; i++)
			data[i][j].write_color(std::cout);
	std::cerr << "图片输出完成, 耗时 " << clock() - t << "ms\n";

	fclose(stdout);
	return 0;
}
