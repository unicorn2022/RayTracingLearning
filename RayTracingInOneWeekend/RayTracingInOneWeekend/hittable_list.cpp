#include "hittable_list.h"
#include "config.h"

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& record) const {
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = t_max; // 获取 ray 相交的最小的 t

	for (const auto& object : objects) {
		if (object->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			record = temp_rec;
		}
	}

	return hit_anything;
}

color hittable_list::ray_color(const ray& r, int depth) {
	hit_record record;

	// 如果碰撞到了, 则根据材质计算反射光线
	if (this->hit(r, 0, INFINITY, record)) {
		ray scattered;
		color attenuation;
		if (depth < max_depth && record._material->scatter(r, record, attenuation, scattered))
			return attenuation * ray_color(scattered, depth + 1);
		else 
			return color(0.0f);
	}
	// 如果不相交, 则根据方向插值背景颜色
	else {
		vec3 direction_unit = r.getDirection().normalize();
		double t = 0.5 * (direction_unit.y() + 1);
		return (1 - t) * color(1.0f) + t * background;
	}
}