#include "material.h"

bool material_diffuse::scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const {
	// 漫反射: 在单位圆内随机取一点, 作为反射方向
	vec3 target = record.position + record.normal + Random::random_unit_sphere();
	r_out = ray(record.position, target - record.position);
	attenuation = albedo;
	return true;
}

bool material_metal::scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const {
	// 模糊镜面反射: 根据法线进行反射, 再加上一个随机扰动
	vec3 target = r_in.getDirection().normalize().reflect(record.normal);
	r_out = ray(record.position, target + fuzz * Random::random_unit_sphere());
	attenuation = albedo;
	return dot(r_out.getDirection(), record.normal) != 0;
}


bool material_dielectric::scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const {
	double eta;				// 折射率
	double reflect_prob;	// 反射概率
	double cos_in;			// 入射角的余弦值	
	vec3 outward_normal;	// 外部法线
	vec3 refracted;			// 折射光线
	vec3 reflected;			// 反射光线
	reflected = r_in.getDirection().reflect(record.normal);
	
	// 衰减永远是1, 因为玻璃不会吸收光线
	attenuation = vec3(1, 1, 1);


	// 入射光线与法线同方向, 说明是从介质中出来的光线
	if (dot(r_in.getDirection(), record.normal) > 0) {
		outward_normal = -record.normal;
		eta = refractive_index;
		cos_in = refractive_index * dot(r_in.getDirection(), record.normal) / r_in.getDirection().length();
	}
	else {
		outward_normal = record.normal;
		eta = 1.0 / refractive_index;
		cos_in = -dot(r_in.getDirection(), record.normal) / r_in.getDirection().length();
	}
	
	// 入射光线发生折射, 计算反射概率
	if (r_in.getDirection().refract(outward_normal, eta, refracted))
		reflect_prob = schlick(cos_in, refractive_index);
	// 入射光线发生全反射, 则反射概率为1
	else
		reflect_prob = 1.0;

	// 根据反射概率, 随机选择反射或折射
	if (Random::random_double_01() < reflect_prob)
		r_out = ray(record.position, reflected);
	else
		r_out = ray(record.position, refracted);

	return true;
}

inline double material_dielectric::schlick(const double cosine, const double refractive_index) const {
	double r0 = (1 - refractive_index) / (1 + refractive_index);
	r0 *= r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}
