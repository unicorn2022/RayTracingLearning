#pragma once

#include <memory>
#include "ray.h"
#include "material.h"

class material;

struct hit_record {
	double t;			// ray 到碰撞点时 t 的大小
	point3 position;	// 碰撞点坐标
	vec3 normal;		// 碰撞点法线
	std::shared_ptr<material> _material;// 碰撞点材质


	// 保证 ray 的方向与法线方向相反
	inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		// ray 的方向与 outward_normal 的方向相反, 需要反转 normal
		bool front_face = dot(r.getDirection(), outward_normal) < 0.0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param record 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& record) const = 0;
};