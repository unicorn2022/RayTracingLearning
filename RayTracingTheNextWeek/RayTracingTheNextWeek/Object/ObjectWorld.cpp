#include "ObjectWorld.h"
#include "../config.h"

Color ObjectWorld::GetColor(const Ray& r, int depth) {
	HitInfo record;

	// 如果碰撞到了, 则根据材质计算反射光线
	if (this->hit(r, 0, INFINITY, record)) {
		Ray scattered;
		Color attenuation;
		if (depth < max_depth && record.material->scatter(r, record, attenuation, scattered))
			return attenuation * GetColor(scattered, depth + 1);
		else
			return Color(0.0f);
	}
	// 如果不相交, 则根据方向插值背景颜色
	else {
		Vec3 direction_unit = r.Direction().normalize();
		double t = 0.5 * (direction_unit.y() + 1);
		return (1 - t) * Color(1.0f) + t * background;
	}
}

bool ObjectWorld::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	HitInfo temp_info;
	bool hit_anything = false;
	double closest_so_far = t_max; // 获取 ray 相交的最小的 t

	for (const auto& object : objects) {
		if (object->hit(r, t_min, closest_so_far, temp_info)) {
			hit_anything = true;
			closest_so_far = temp_info.t;
			info = temp_info;
		}
	}

	return hit_anything;
}
