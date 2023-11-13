#pragma once

#include "../Math/Ray.h"
#include "../Object/Object.h"

class HitInfo;

class Material {
public:
	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param info 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	virtual bool scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const = 0;

	/*
	* @brief 自发光
	* @param u uv坐标
	* @param v uv坐标
	* @param p 碰撞点
	*/
	virtual Color emitted(double u, double v, const Point3& p) const {
		return Color(0, 0, 0);
	}
};