#pragma once

#include "ObjectBase.h"
#include "../Utils/Utils.h"
#include "../Material/MaterialBase.h"


class Sphere : public ObjectBase {
public:
	Sphere(Point3 c, double r, Ref<MaterialBase> m) : center(c), radius(r), material(m) {}

	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param info 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const Ray& r, double t_min, double t_max, HitInfo& info) const override;

private:
	Point3 center;
	double radius;
	Ref<MaterialBase> material;
};

