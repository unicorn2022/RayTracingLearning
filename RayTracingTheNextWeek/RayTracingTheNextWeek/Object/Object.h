#pragma once

#include "../config.h"
#include "../Math/Ray.h"
#include "../Material/Material.h"
#include "AABB.h"

class Material;

class HitInfo {
public:
	double t;				// ray 到碰撞点时 t 的大小
	Point3 position;		// 碰撞点坐标
	Vec3 normal;			// 碰撞点法线
	Ref<Material> material;	// 碰撞点材质
	double u, v;			// 碰撞点 uv 坐标
};

class Object {
public:
	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param info 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const Ray& r, double t_min, double t_max, HitInfo& info) const = 0;

	/*
	* @brief 获取当前对象的包围盒
	*/
	virtual AABB GetBox() const = 0;
};

