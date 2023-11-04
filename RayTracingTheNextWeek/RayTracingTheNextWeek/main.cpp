#pragma warning(disable:4996)
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

/* 全局参数设置 */
namespace {
	/* 图片设置 */
	const int Image_Width = 400;			// 图片宽度
	const int Image_Height = 225;			// 图片高度
	Color data[Image_Width][Image_Height];	// 图片数据
	int total_completed = 0;				// 已完成的列数
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

void Render(int L, int R, bool single) {
	for (int j = L; j < R; j++) {
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

		total_completed++;
		if(single) std::cerr << "\r" << "已完成: " << total_completed << "/" << Image_Height << ", (" << total_completed * 100 / Image_Height << "%)" << std::flush;
	}
}

int main() {
	freopen("../output.ppm", "w", stdout);
	
	auto t = clock();
	AddObjects();

	// 渲染图片
	if (thread_cnt == 1) {
		std::cerr << "单线程模式\n";
		Render(0, Image_Height, true);
	}
	else {
		std::cerr << "多线程模式, 线程数: " << thread_cnt << "\n";
		int num = Image_Height / thread_cnt;
		for (int i = 0; i < thread_cnt; i++) {
			int L = i * num, R = (i + 1) * num;
			if(i == thread_cnt - 1) R = Image_Height;

			std::thread ti(Render, L, R, false);
			ti.detach();
		}
	
		// 监视渲染进度
		while (total_completed < Image_Height) {
			std::cerr << "\r" << "已完成: " << total_completed << "/" << Image_Height << ", (" <<  total_completed * 100 / Image_Height << "%)" << std::flush;
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
		std::cerr << "\r" << "已完成: " << Image_Height << "/" << Image_Height << ", (100%)" << std::flush;
	}
	std::cerr << "\n图片渲染完成, 耗时 " << (clock() - t) / 1000.0f << "s\n";

	// 输出图片
	t = clock();
	std::cout << "P3\n" << Image_Width << " " << Image_Height << "\n255\n";
	for (int j = Image_Height - 1; j >= 0; j--)
		for (int i = 0; i < Image_Width; i++)
			data[i][j].write_color(std::cout);
	std::cerr << "图片输出完成, 耗时 " << (clock() - t) / 1000.0f << "s\n";

	fclose(stdout);
	return 0;
}
