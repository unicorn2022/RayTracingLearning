#include "Lambertian.h"
#include "../Math/Random.h"


bool Lambertian::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
	// 漫反射: 在单位圆内随机取一点, 作为反射方向
	Vec3 target = info.position + info.normal + Random::rand_unit_sphere();
	r_out = Ray(info.position, target - info.position);
	attenuation = albedo->Value(info.u, info.v, info.position);
	return true;
}
