#pragma warning(disable:4996)
#include "main.h"

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
	Vec3 from(278, 278, -800);
	Vec3 at(278, 278, 0);
	double dist_to_focus = 10;
	double aperture = 0.0;
	double vfov = 40.0;
	double time_start = 0, time_end = 1.0;
	Camera main_camera(from, at, Vec3(0, 1, 0), vfov, aspect, aperture, 0.7 * dist_to_focus, time_start, time_end);	// 主相机
}

//int AABB_hit = 0;
//int BVH_leaf_hit = 0;
//int BVH_node_cnt = 0;

void AddObjects() {
	Ref<Material> red = New<Lambertian>(New<TextureConstant>(Color(0.65, 0.05, 0.05)));
	Ref<Material> white = New<Lambertian>(New<TextureConstant>(Color(0.73, 0.73, 0.73)));
	Ref<Material> green = New<Lambertian>(New<TextureConstant>(Color(0.12, 0.45, 0.15)));
	Ref<Material> light = New<Emit>(New<TextureConstant>(Color(20, 20, 20)));

	world.Add(New<FlipNormal>(New<RectYZ>(0, 555, 0, 555, 555, green)));	// 左绿墙
	world.Add(New<RectYZ>(0, 555, 0, 555, 0, red));		// 右红墙
	world.Add(New<RectXZ>(213, 343, 227, 332, 554, light));
	world.Add(New<FlipNormal>(New<RectXZ>(0, 555, 0, 555, 555, white)));	// 上白墙
	world.Add(New<RectXZ>(0, 555, 0, 555, 0, white));	// 下白墙
	world.Add(New<FlipNormal>(New<RectXY>(0, 555, 0, 555, 555, white)));	// 后白墙
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
				int depth = 0;
				data[i][j] += world.GetColor(r, depth);
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
