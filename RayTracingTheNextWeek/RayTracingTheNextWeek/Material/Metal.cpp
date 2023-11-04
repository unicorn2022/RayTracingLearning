#include "Metal.h"

bool Metal::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
	// 模糊镜面反射: 根据法线进行反射, 再加上一个随机扰动
	Vec3 target = r_in.Direction().normalize().reflect(info.normal);
	r_out = Ray(info.position, target + fuzz * Random::random_unit_sphere());
	attenuation = albedo;
	return r_out.Direction().dot(info.normal) != 0;
}
