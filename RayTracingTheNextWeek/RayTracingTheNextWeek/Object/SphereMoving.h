#pragma once
#include "Object.h"
#include "../Utils/Utils.h"
#include "../Material/Material.h"

class SphereMoving : public Object {
public:
	/*
	* @param center1	time1时的球心坐标
	* @param center2	time2时的球心坐标
	* @param time1		球心坐标的时间1
	* @param time2		球心坐标的时间2
	* @param radius		半径
	* @param material	材质
	*/
	SphereMoving(Point3 center1, Point3 center2, double time1, double time2, double radius, Ref<Material> material)
		: center1(center1), center2(center2), time1(time1), time2(time2), radius(radius), material(material) {}

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

	/*
	* @brief 获取 time 时刻的球心坐标
	* @param time 时间
	*/
	Point3 GetCenter(double time) const;

private:
	Point3 center1;
	Point3 center2;
	double time1;
	double time2;
	double radius;
	Ref<Material> material;
};

