#pragma once

#include "Object.h"
#include "../config.h"
#include "../Material/Material.h"


class Sphere : public Object {
public:
	Sphere(Point3 c, double r, Ref<Material> m) : center(c), radius(r), material(m) {}

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
	/*
	* @brief 获取碰撞点的 uv 坐标
	* @param p 碰撞点的局部坐标(归一化)
	* @param u uv 坐标
	* @param v uv 坐标
	*/
	void GetUV(const Point3& p, double& u, double& v) const;

	Point3 center;
	double radius;
	Ref<Material> material;
};

