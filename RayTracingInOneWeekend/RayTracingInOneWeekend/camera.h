#pragma once

#include "vec3.h"
#include "ray.h"

class camera {
public:
	/*
	* @brief 透视投影相机
	* @param lookfrom	相机位置, 默认为原点(0, 0, 0)
	* @param lookat		相机看向的位置, 默认为屏幕中心(0, 0, -1)
	* @param vup		相机向上的方向, 默认为y轴正方向(0, 1, 0)
	* @param vfov		垂直方向视角
	* @param aspect		宽高比
	* @param aperture	光圈直径
	* @param focus		焦距
	*/
	camera(vec3 lookfrom, vec3 lookat, vec3 vup, double vfov, double aspect, double aperture, double focus)
		: position(lookfrom), lens_radius(aperture/2) {
		double theta = vfov * std::_Pi / 180;
		double half_height = tan(theta / 2) * focus; // tan(θ/2) = (h/2) / 焦距
		double half_width = aspect * half_height;

		// 相机坐标系
		w = (lookfrom - lookat).normalize();	// 指向相机
		u = cross(vup, w).normalize();			// 指向屏幕右侧
		v = cross(w, u).normalize();			// 指向屏幕上方

		// 相机向量
		low_left_corner = position - half_width * u - half_height * v - focus * w; // 高和宽都乘了焦距，w也要乘，不然公式是错的
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
	}

	/*
	* @brief 获取当前像素对应的光线
	* @param u	当前像素的[0,1]坐标
	* @param v	当前像素的[0,1]坐标
	*/
	inline ray getRay(double u, double v) const {
		// 光圈随机偏移
		vec3 rd = lens_radius * Random::random_unit_sphere();
		vec3 offset = u * rd.x() + v * rd.y();
		
		// 光线的起点为原点, 方向指向观察平面上的当前像素
		vec3 start = position + offset;
		vec3 target = low_left_corner + u * horizontal + v * vertical;
		return ray(start, target - start);
	}

public:
	inline const vec3& getPosition() const { return position; }
	inline void setPosition(const vec3& position) { this->position = position; }
	inline const point3& getLowLeftCorner() const { return low_left_corner; }
	inline const vec3& getHorizontal() const { return horizontal; }
	inline const vec3& getVertical() const { return vertical; }

private:
	vec3 u, v, w;	// 相机坐标系

	vec3 position;			// 相机位置
	point3 low_left_corner;	// 屏幕左下角坐标
	vec3 horizontal;		// 屏幕宽度向量: x轴
	vec3 vertical;			// 屏幕高度向量: y轴
	double lens_radius;		// 光圈半径
};