#pragma once

#include "../Math/Ray.h"

class AABB {
public:
	AABB() :min(Vec3(-1e5)), max(Vec3(1e5)) {}

	/*
	* @param min 包围盒的最小坐标
	* @param max 包围盒的最大坐标
	*/
	AABB(const Vec3& min, const Vec3& max) : min(min), max(max) {}
	
	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @return 是否碰撞
	*/
	bool hit(const Ray& r, double t_min, double t_max) const;

	static AABB Merge(const AABB& box1, const AABB& box2);

public:
	Vec3 Min() const { return min; }
	Vec3 Max() const { return max; }

private:
	Vec3 min;
	Vec3 max;
};

