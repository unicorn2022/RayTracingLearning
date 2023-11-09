#include "AABB.h"

bool AABB::hit(const Ray& r, double t_min, double t_max) const {
	//static int cnt = 0;
	//std::cerr << "AABB hit: " << ++cnt << "\n";

	// 计算XYZ三个轴的 t 值
	// t1 = (min.x - origin.x) / direction.x
	// t2 = (max.x - origin.x) / direction.x
	for (int i = 0; i < 3; i++) {
		double div = 1.0 / r.Direction()[i];
		double t1 = (min[i] - r.Origin()[i]) / r.Direction()[i];
		double t2 = (max[i] - r.Origin()[i]) / r.Direction()[i];
		if(div < 0) std::swap(t1, t2);
		if (std::min(t2, t_max) <= std::max(t1, t_min)) return false;
	}
	return true;
}

AABB AABB::Merge(const AABB& box1, const AABB& box2) {
	Vec3 Min{
		std::min(box1.Min().x(), box2.Min().x()),
		std::min(box1.Min().y(), box2.Min().y()),
		std::min(box1.Min().z(), box2.Min().z())
	};
	Vec3 Max{
		std::max(box1.Max().x(), box2.Max().x()),
		std::max(box1.Max().y(), box2.Max().y()),
		std::max(box1.Max().z(), box2.Max().z())
	};
	return AABB(Min, Max);
}
