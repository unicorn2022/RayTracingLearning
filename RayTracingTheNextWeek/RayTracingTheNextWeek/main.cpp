#pragma warning(disable:4996)
#include <iostream>
#include <iomanip>
#include <ctime>
#include <thread>

#include "Camera/Camera.h"
#include "Math/Random.h"
#include "Object/Sphere.h"
#include "Object/SphereMoving.h"
#include "Object/ObjectWorld.h"
#include "Material/Lambertian.h"
#include "Material/Dielectric.h"
#include "Material/Metal.h"
#include "Texture/TextureConstant.h"
#include "Texture/TextureChecker.h"
#include "Texture/TextureNoise.h"
#include "Texture/TextureImage.h"
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
	Vec3 from(13, 2, 3);
	Vec3 at(0, 0, 0);
	double dist_to_focus = 10;
	double aperture = 0.0;
	double time_start = 0, time_end = 1.0;
	Camera main_camera(from, at, Vec3(0, 1, 0), 20, aspect, aperture, 0.7 * dist_to_focus, time_start, time_end);	// 主相机
}

//int AABB_hit = 0;
//int BVH_leaf_hit = 0;
//int BVH_node_cnt = 0;

void AddObjects() {
	// 地面
	world.Add(New<Sphere>(
		Point3(0, -1000, 0), 
		999.2, 
		New<Metal>(New<TextureConstant>(Color(0.8, 0.2, 0.0)), 0.0))
	);	
	// 物体
	world.Add(New<Sphere>(
		Point3(0, 0, 0),
		0.8,
		New<Lambertian>(New<TextureImage>("resource/earthmap.jpg")))
	);
	return;
}

void Render(int L, int R, bool single) {
	for (int j = L; j < R; j++) {
		for (int i = 0; i < Image_Width; i++) {
			// 每个像素随机采样 samples_per_pixel 次, 并取平均值
			for (int cnt = 0; cnt < samples_per_pixel; cnt++) {
				// 当前像素的[0,1]坐标
				auto u = double(i + Random::rand01()) / Image_Width;
				auto v = double(j + Random::rand01()) / Image_Height;

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
		for (int i = 0; i < Image_Width; i++) {
			Color now = data[i][j];
			// 抗锯齿
			if (i + 1 < Image_Width && j + 1 < Image_Height) {
				now = (data[i][j] + data[i + 1][j] + data[i][j + 1] + data[i + 1][j + 1]) / 4;
			}
			now.write_color(std::cout);
		}
	std::cerr << "图片输出完成, 耗时 " << (clock() - t) / 1000.0f << "s\n";

	fclose(stdout);
	return 0;
}
