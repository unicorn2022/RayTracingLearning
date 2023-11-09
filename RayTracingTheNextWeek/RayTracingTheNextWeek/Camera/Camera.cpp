#include "Camera.h"

Camera::Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus, double t1, double t2)
	: position(lookfrom), lens_radius(aperture / 2), time1(t1), time2(t2) {
	double theta = vfov * std::_Pi / 180;
	double half_height = tan(theta / 2) * focus; // tan(θ/2) = (h/2) / 焦距
	double half_width = aspect * half_height;

	// 相机坐标系
	w = (lookfrom - lookat).normalize();// 指向相机
	u = vup.cross(w).normalize();		// 指向屏幕右侧
	v = w.cross(u).normalize();			// 指向屏幕上方

	// 相机向量
	low_left_corner = position - half_width * u - half_height * v - focus * w; // 高和宽都乘了焦距，w也要乘，不然公式是错的
	horizontal = 2 * half_width * u;
	vertical = 2 * half_height * v;
}

Ray Camera::GetRay(double u, double v) const {
	// 光圈随机偏移
	Vec3 rd = lens_radius * Random::rand_unit_sphere();
	Vec3 offset = u * rd.x() + v * rd.y();

	// 光线的起点为原点, 方向指向观察平面上的当前像素
	// 当前时间为 [time1, time2] 之间的随机值
	Vec3 start = position + offset;
	Vec3 target = low_left_corner + u * horizontal + v * vertical;
	double time = time1 + Random::rand01() * (time2 - time1);
	return Ray(start, target - start, time);
}
