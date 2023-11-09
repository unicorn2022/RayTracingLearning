#include "Dielectric.h"

bool Dielectric::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
	double eta;				// 折射率
	double reflect_prob;	// 反射概率
	double cos_in;			// 入射角的余弦值	
	Vec3 outward_normal;	// 外部法线
	Vec3 refracted;			// 折射光线
	Vec3 reflected;			// 反射光线
	reflected = r_in.Direction().reflect(info.normal);

	// 衰减永远是1, 因为玻璃不会吸收光线
	attenuation = Vec3(1, 1, 1);


	// 入射光线与法线同方向, 说明是从介质中出来的光线
	if (r_in.Direction().dot(info.normal) > 0) {
		outward_normal = -info.normal;
		eta = refractive_index;
		cos_in = refractive_index * r_in.Direction().dot(info.normal) / r_in.Direction().length();
	}
	else {
		outward_normal = info.normal;
		eta = 1.0 / refractive_index;
		cos_in = -r_in.Direction().dot(info.normal) / r_in.Direction().length();
	}

	// 入射光线发生折射, 计算反射概率
	if (r_in.Direction().refract(outward_normal, eta, refracted))
		reflect_prob = schlick(cos_in, refractive_index);
	// 入射光线发生全反射, 则反射概率为1
	else
		reflect_prob = 1.0;

	// 根据反射概率, 随机选择反射或折射
	if (Random::rand01() < reflect_prob)
		r_out = Ray(info.position, reflected);
	else
		r_out = Ray(info.position, refracted);

	return true;
}

double Dielectric::schlick(const double cosine, const double refractive_index) const {
	double r0 = (1 - refractive_index) / (1 + refractive_index);
	r0 *= r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}
