﻿#pragma once
#include "../Object.h"
class RotateY : public Object {
public:
	RotateY(double angle, Ref<Object> object);

	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param info 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const Ray& r, double t_min, double t_max, HitInfo& info) const override;

	/*
	* @brief 获取当前对象的包围盒
	*/
	virtual AABB GetBox() const override;
private:

	Ref<Object> object;
	double sin_theta, cos_theta;
	AABB box;
};

