#include "Lambertian.h"
#include "../Math/Random.h"
#include "../Math/ONB.h"


bool Lambertian::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
	double pdf;
	return scatter(r_in, info, attenuation, r_out, pdf);
}

bool Lambertian::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out, double& pdf) const {
	ONB uvw(info.normal);
	do {
		// 在单位球上随机采样方向, 作为出射方向, 然后将其转化为全局坐标
		Vec3 direction = uvw.local(Random::rand_cosine_direction());
		
		// 构建出射光线
		r_out = Ray(info.position, direction.normalize(), r_in.Time());
		
		// 重要性采样 pdf = cosθ / π
		pdf = uvw.w().dot(r_out.Direction()) / PI;
	} while (pdf == 0);

	attenuation = albedo->Value(info.u, info.v, info.position);
	return true;
}

double Lambertian::scatter_pdf(const Ray& r_in, const HitInfo& info, const Ray& r_out) const {
	double cosine = info.normal.dot(r_out.Direction());
	if (cosine < 0) cosine = 0;
	return cosine / PI;
}
