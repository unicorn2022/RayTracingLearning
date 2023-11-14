#include "SphereMoving.h"
#include <random>
static const double PI = std::_Pi;

bool SphereMoving::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	// 根据公式判断是否相交
	// (1) 球	: (P - center)^2 = radius^2
	// (2) 光线	: P = origin + t * direction
	// 带入求t: t^2 * direction^2 + 2t * direction * (origin - center) + (origin - center)^2 - radius^2 = 0
	Point3 center = GetCenter(r.Time());
	Vec3 oc = r.Origin() - center;
	auto a = r.Direction().dot(r.Direction());
	auto b = 2.0 * oc.dot(r.Direction());
	auto c = oc.dot(oc) - radius * radius;
	auto discrimination = b * b - 4 * a * c;

	if (discrimination < 0) return false;

	auto sqrtd = sqrt(discrimination);
	auto t = (-b - sqrtd) / (2 * a);
	if (t < t_min || t > t_max) {
		t = (-b + sqrtd) / (2 * a);
		if (t < t_min || t > t_max) return false;
	}

	info.t = t;
	info.position = r.At(t);
	info.normal = (info.position - center) / radius; // 法线: 球心指向相交点
	info.material = material;
	GetUV((info.position - center) / radius, info.u, info.v); // 计算uv(球心为原点)

	return true;
}

AABB SphereMoving::GetBox() const {
	AABB A(
		center1 - Vec3(radius, radius, radius),
		center1 + Vec3(radius, radius, radius)
	);
	AABB B(
		center2 - Vec3(radius, radius, radius),
		center2 + Vec3(radius, radius, radius)
	);
	return AABB::Merge(A, B);
}

Point3 SphereMoving::GetCenter(double time) const {
    return center1 + ((time - time1) / (time2 - time1)) * (center2 - center1);
}

void SphereMoving::GetUV(const Point3& p, double& u, double& v) const {
	// x = cos(theta) * cos(phi)
	// z = cos(theta) * sin(phi)
	// y = sin(theta)
	double phi = atan2(p.z(), p.x()) + PI;
	double theta = asin(p.y()) + PI / 2;
	u = phi / (2 * PI);
	v = theta / PI;
}
