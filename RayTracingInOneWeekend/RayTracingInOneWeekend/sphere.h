#pragma once

#include <memory>
#include "hittable.h"
#include "material.h"

class sphere : public hittable{
public:
	sphere() : center(0.0f), radius(0.0f) {}
	sphere(point3 c, double r, std::shared_ptr<material> m) : center(c), radius(r), _material(m) {}
	
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& record) const override;

private:
	point3 center;
	double radius;
	std::shared_ptr<material> _material;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& record) const {
	// 根据公式判断是否相交
	// (1) 球	: (P - center)^2 = radius^2
	// (2) 光线	: P = origin + t * direction
	// 带入求t: t^2 * direction^2 + 2t * direction * (origin - center) + (origin - center)^2 - radius^2 = 0
	vec3 oc = r.getOrigin() - center;
	auto a = dot(r.getDirection(), r.getDirection());
	auto b = 2.0 * dot(oc, r.getDirection());
	auto c = dot(oc, oc) - radius * radius;
	auto discrimination = b * b - 4 * a * c;

	if (discrimination < 0) return false;

	auto sqrtd = sqrt(discrimination);
	auto t = (-b - sqrtd) / (2 * a);
	if (t < t_min || t > t_max) {
		t = (-b + sqrtd) / (2 * a);
		if (t < t_min || t > t_max) return false;
	}

	record.t = t;
	record.position = r.at(t);
	record._material = _material;
	vec3 outward_normal = (record.position - center) / radius; // 法线: 球心指向相交点
	record.set_face_normal(r, outward_normal);
	
	return true;
}