#pragma once

#include "../Utils/Utils.h"
#include "../Math/Vec3.h"
#include "../Math/Ray.h"

class Camera {
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
	* @param t1			快门开启时间
	* @param t2			快门关闭时间
	*/
	Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, double vfov, double aspect, double aperture, double focus, double t1, double t2);

	/*
	* @brief 获取当前像素对应的光线
	* @param u	当前像素的[0,1]坐标
	* @param v	当前像素的[0,1]坐标
	*/
	Ray GetRay(double u, double v) const;

private:
	Vec3 u, v, w;	// 相机坐标系

	Vec3 position;			// 相机位置
	Point3 low_left_corner;	// 屏幕左下角坐标
	Vec3 horizontal;		// 屏幕宽度向量: x轴
	Vec3 vertical;			// 屏幕高度向量: y轴
	double lens_radius;		// 光圈半径
	double time1;			// 快门开启时间
	double time2;			// 快门关闭时间
};
