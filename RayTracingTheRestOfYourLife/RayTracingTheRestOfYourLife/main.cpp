#pragma warning(disable:4996)
#include "main.h"

/* 全局参数设置 */
namespace {
	/* 图片设置 */
	const int Image_Width = 400;			// 图片宽度
	const int Image_Height = 225;			// 图片高度
	Color data[Image_Width][Image_Height];	// 图片数据
	const int Image_pixel = Image_Width * Image_Height;
	const double aspect = 1.0f * Image_Width / Image_Height;	// 宽高比
	int pixel[Image_Width * Image_Height];

	/* 线程监控 */
	int total_completed = 0;	// 已完成的像素数
	int completed[thread_cnt];	// 每个线程已完成的像素数
	int total[thread_cnt];		// 每个线程总共需要完成的像素数

	/* 世界设置 */
	Color background(0.5, 0.7, 1.0);// 背景颜色
	ObjectWorld world(background);	// 世界中的物体

	/* 相机设置 */
	Vec3 from(300, 278, -520);
	Vec3 at(278, 278, 0);
	double dist_to_focus = (from - at).length();
	double aperture = 0.0;
	double vfov = 75.0;
	double time_start = 0, time_end = 1.0;
	Camera main_camera(from, at, Vec3(0, 1, 0), vfov, aspect, aperture, 0.7 * dist_to_focus, time_start, time_end);	// 主相机
}

//int AABB_hit = 0;
//int BVH_leaf_hit = 0;
//int BVH_node_cnt = 0;

void Cornell_smoke() {
	Ref<Material> red = New<Lambertian>(New<TextureConstant>(Color(0.65, 0.05, 0.05)));
	Ref<Material> blue = New<Lambertian>(New<TextureConstant>(Color(0.05, 0.05, 0.73)));
	Ref<Material> green = New<Lambertian>(New<TextureConstant>(Color(0.12, 0.45, 0.15)));
	Ref<Material> white = New<Lambertian>(New<TextureConstant>(Color(0.88, 0.88, 0.88)));
	Ref<Material> light = New<Emit>(New<TextureConstant>(Color(1, 1, 1)));

	world.Add(New<RectXZ>(213, 343, 227, 332, 554, light));				// 顶部光源
	world.Add(New<RectYZ>(0, 555, 0, 555, 555, green));					// 左绿墙
	world.Add(New<FlipNormal>(New<RectYZ>(0, 555, 0, 555, 0, red)));	// 右红墙
	world.Add(New<FlipNormal>(New<RectXZ>(0, 555, 0, 555, 555, white)));// 上白墙
	world.Add(New<RectXZ>(0, 555, 0, 555, 0, white));					// 下白墙
	world.Add(New<FlipNormal>(New<RectXY>(0, 555, 0, 555, 555, blue)));	// 后蓝墙

	Ref<Box> box1 = New<Box>(Vec3(0, 0, 0), Vec3(165, 165, 165), white);
	Ref<Box> box2 = New<Box>(Vec3(0, 0, 0), Vec3(165, 330, 165), white);
	Ref<Object> box_1 = New<Translate>(Vec3(130, 0, 65), New<RotateY>(-18, box1));
	Ref<Object> box_2 = New<Translate>(Vec3(265, 0, 295), New<RotateY>(15, box2));
	world.Add(New<ConstantMedium>(box_2, 0.006, New<TextureConstant>(Color(0.8, 0.58, 0))));	// 盒子1
	world.Add(New<ConstantMedium>(box_1, 0.008, New<TextureConstant>(Color(0.9, 0.2, 0.72))));	// 盒子2
	return;
}

void Final_Scene() {
	Ref<Material> red = New<Lambertian>(New<TextureConstant>(Color(0.65, 0.05, 0.05)));
	Ref<Material> blue = New<Lambertian>(New<TextureConstant>(Color(0.05, 0.05, 0.73)));
	Ref<Material> green = New<Lambertian>(New<TextureConstant>(Color(0.12, 0.45, 0.15)));
	Ref<Material> white = New<Lambertian>(New<TextureConstant>(Color(0.88, 0.88, 0.88)));
	Ref<Material> ground = New<Lambertian>(New<TextureConstant>(Color(0.48, 0.83, 0.53)));
	Ref<Material> light = New<Emit>(New<TextureConstant>(Color(1, 1, 1)));

	// 20 * 20 个盒子
	Ref<ObjectWorld> boxes = New<ObjectWorld>();
	int nb = 20;
	for(int i = 0; i < nb; i++)
		for (int j = 0; j < nb; j++) {
			double w = 100;
			double x0 = -1000 + i * w;
			double z0 = -1000 + j * w;
			double y0 = 0;
			double x1 = x0 + w;
			double y1 = 100 * (Random::rand01() + 0.01);
			double z1 = z0 + w;
			boxes->Add(New<Box>(Vec3(x0, y0, z0), Vec3(x1, y1, z1), ground));
		}
	boxes->Build();
	world.Add(boxes);

	// 顶部光源
	world.Add(New<RectXZ>(123, 423, 147, 412, 550, light));

	// 2个静止球 + 1个运动球
	Vec3 center(400, 400, 200);
	world.Add(New<SphereMoving>(center, center + Vec3(30, 0, 0), 0, 1, 50, New<Lambertian>(New<TextureConstant>(Color(0.7, 0.3, 0.1)))));
	world.Add(New<Sphere>(Vec3(260, 150, 45), 50, New<Dielectric>(1.5)));
	world.Add(New<Sphere>(Vec3(0, 150, 145), 50, New<Metal>(New<TextureConstant>(Color(0.8, 0.8, 0.9)), 10.0)));

	// 体积雾
	Ref<Object> boundary = New<Sphere>(Vec3(360, 150, 145), 70, New<Dielectric>(1.5));
	world.Add(boundary);
	world.Add(New<ConstantMedium>(boundary, 0.2, New<TextureConstant>(Color(0.2, 0.4, 0.9))));
	boundary = New<Sphere>(Vec3(0, 0, 0), 5000, New<Dielectric>(1.5));
	world.Add(New<ConstantMedium>(boundary, 0.0001, New<TextureConstant>(Color(1, 1, 1))));

	// 贴图球
	Ref<Material> earth = New<Lambertian>(New<TextureImage>("resource/earthmap.jpg"));
	world.Add(New<Sphere>(Vec3(400, 200, 400), 100, earth));

	// 噪声球
	Ref<Material> noise = New<Lambertian>(New<TextureNoise>(0.1));
	world.Add(New<Sphere>(Vec3(220, 280, 300), 80, noise));

	// 1000 个球
	int ns = 1000;
	Ref<ObjectWorld> spheres = New<ObjectWorld>();
	for(int j = 0; j < ns; j++)
		spheres->Add(New<Sphere>(Vec3(165 * Random::rand01(), 165 * Random::rand01(), 165 * Random::rand01()), 10, white));
	spheres->Build();
	world.Add(New<Translate>(Vec3(-100, 270, 395), New<RotateY>(15, spheres)));
	world.Build();
}

void Render(int L, int R, bool single, int number) {
	for (int k = L; k < R; k++) {
		int i = pixel[k] / Image_Height, j = pixel[k] % Image_Height;
		// 每个像素随机采样 samples_per_pixel 次, 并取平均值
		for (int cnt = 0; cnt < samples_per_pixel; cnt++) {
			// 当前像素的[0,1]坐标
			auto u = double(i + Random::rand01()) / Image_Width;
			auto v = double(j + Random::rand01()) / Image_Height;

			// 获取当前像素对应的光线
			Ray r = main_camera.GetRay(u, v);
			//if (u >= 0.24 && u <= 0.26 && v >= 0.49 && v <= 0.51)
			//	int aaa = 1;
			// 计算光线得到的颜色
			int depth = 0;
			data[i][j] += world.GetColor(r, depth);
		}
		data[i][j] /= samples_per_pixel;

		// gamma 矫正
		data[i][j].gamma();

		completed[number]++;

		if (single) {
			std::cout << "\r" << "已完成: " << total_completed << "/" << Image_pixel << ", ";
			PrintPercent(total_completed, Image_pixel);
			std::cout << std::flush;
		}
	}
}

void RenderPicture() {
	auto t = clock();
	Final_Scene();

	// 将图片的像素编号并打乱, 从而能够随机分给不同线程
	for (int i = 0; i < Image_pixel; i++)
		pixel[i] = i;
	std::shuffle(pixel, pixel + Image_pixel, std::mt19937(std::random_device()()));

	// 渲染图片
	if (thread_cnt == 1) {
		std::cout << "单线程模式\n";
		Render(0, Image_pixel, true, 0);
	}
	else {
		std::cout << "多线程模式, 线程数: " << thread_cnt << "\n";
		int num = Image_pixel / thread_cnt;
		for (int i = 0; i < thread_cnt; i++) {
			int L = i * num, R = (i + 1) * num;
			if (i == thread_cnt - 1) R = Image_pixel;

			total[i] = R - L;
			std::thread ti(Render, L, R, false, i);
			ti.detach();
		}

		// 监视渲染进度
		while(true) {
			total_completed = 0;
			for (int i = 0; i < thread_cnt; i++) total_completed += completed[i];

			std::cout << "\r" << "已完成:" << SetConsoleColor(ConsoleColor::Pink) << total_completed << SetConsoleColor(ConsoleColor::Clear) << "/" << Image_pixel << ", ";
			std::cout << "预计剩余时间:" << SetConsoleColor(ConsoleColor::Cyan) << ceil((clock() - t) / 1000.0f / total_completed * (Image_pixel - total_completed)) << SetConsoleColor(ConsoleColor::Clear) << "s ";

			for (int i = 0; i < thread_cnt; i++) {
				SetConsoleColor(ConsoleColor::Yellow);
				std::cout << i << ":";
				SetConsoleColor(ConsoleColor::Clear);

				PrintPercent(completed[i], total[i]);
			}

			std::cout << std::flush;
			if (total_completed >= Image_pixel) break;

			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		}
	}
	std::cout << "\n图片渲染完成, 耗时 " << SetConsoleColor(ConsoleColor::Cyan) << (clock() - t) / 1000.0f << SetConsoleColor(ConsoleColor::Clear) << " s\n";
}

void OutputPicture() {
	std::ofstream fout("../output.ppm");

	// 输出图片
	auto t = clock();
	fout << "P3\n" << Image_Width << " " << Image_Height << "\n255\n";
	for (int j = Image_Height - 1; j >= 0; j--)
		for (int i = 0; i < Image_Width; i++)
			data[i][j].write_color(fout);
	std::cout << "图片输出完成, 耗时 " << SetConsoleColor(ConsoleColor::Cyan) << (clock() - t) / 1000.0f << SetConsoleColor(ConsoleColor::Clear) << " s\n";

	fout.close();
}

int main() {
	RenderPicture();
	OutputPicture();
	return 0;
}
