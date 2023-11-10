#pragma once
#include "../Math/Vec3.h"

class Texture {
public:
	/*
	* @brief 获取纹理颜色
	* @param u 纹理坐标u
	* @param v 纹理坐标v
	* @param p 坐标
	* @return 纹理颜色
	*/
	virtual Color Value(double u, double v, const Point3& p) const = 0;
};

