#pragma warning(disable:4996)
#include <iostream>
#include <iomanip>
#include "utils.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "config.h"

/* 全局参数设置 */
namespace {
	/* 图片设置 */
	const double aspect_ratio = 16.0 / 9.0;	// 宽高比
	const int Image_Width = 400;			// 图片宽度
	const int Image_Height = static_cast<int>(Image_Width / aspect_ratio);	// 图片高度

	/* 世界设置 */
	color background(0.5, 0.7, 1.0);	// 背景颜色
	hittable_list world(background);	// 世界中的物体

	/* 相机设置 */
	vec3 from(3, 3, 2);
	vec3 at(0, 0, -1);
	double focus = (from - at).length();
	camera main_camera(from, at, vec3(0,1,0), 20, aspect_ratio, 2.0, focus);	// 主相机
}

/* 添加物体到世界中 */
void add_objects() {
	world.add(std::make_shared<sphere>(
		point3(0, 0, -1),		
		0.5, 
		std::make_shared<material_diffuse>(color(0.1, 0.2, 0.5)))
	);
	world.add(std::make_shared<sphere>(
		point3(0, -100.5, -1),
		100,
		std::make_shared<material_metal>(color(0.8, 0.8, 0), 0.3))
	);
	world.add(std::make_shared<sphere>(
		point3(-1, 0, -1),
		0.5,
		std::make_shared<material_metal>(color(0, 0, 1)))
	);
	world.add(std::make_shared<sphere>(
		point3(1, 0, -1),
		0.5,
		std::make_shared<material_diffuse>(color(1, 0, 0)))
	);
}

int main() {
	freopen("../output.ppm", "w", stdout);

	add_objects();

	// PPM头: P3\n 宽 高\n 最大颜色值\n
	std::cout << "P3\n" << Image_Width << " " << Image_Height << "\n255\n";
	
	// 像素顺序: 从左到右, 从上到下, (0, h-1) => (w-1, 0)
	for (int j = Image_Height - 1; j >= 0; j--) {
		double completed = 100 - (j * 100) / Image_Height;
		std::cerr << "\r已完成: " << std::setw(3) << completed << "%" << std::flush;
		std::cerr << "\t剩余行数: " << std::setw(3) << j << std::flush;
		
		for (int i = 0; i < Image_Width; i++) {
			if (i % 10 == 0) {
				std::cerr << "\r\t\t\t\t当前行进度: " << std::setw(3) << i << "/" << Image_Width << std::flush;
			}
			
			color pixel;
			
			// 每个像素随机采样 samples_per_pixel 次, 并取平均值
			for (int cnt = 0; cnt < samples_per_pixel; cnt++) {
				// 当前像素的[0,1]坐标
				auto u = double(i + Random::random_double_01()) / Image_Width;
				auto v = double(j + Random::random_double_01()) / Image_Height;

				// 获取当前像素对应的光线
				ray r = main_camera.getRay(u, v);

				// 计算光线得到的颜色
				pixel += world.ray_color(r);
			}
			pixel /= samples_per_pixel;
			
			// gamma 矫正
			pixel.gamma();

			// 输出颜色
			pixel.write_color(std::cout);
		}
	}

	std::cerr << "\nDone!\n";
	fclose(stdin);
	return 0;
}