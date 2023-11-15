#include "RotateY.h"
#include <random>
#include <cmath>
static const double PI = std::_Pi;

RotateY::RotateY(double angle, Ref<Object> object) : object(object) {
	double radians = angle * PI / 180.0;
	sin_theta = std::sin(radians);
	cos_theta = std::cos(radians);

	// 计算包围盒
	box = object->GetBox();
	Vec3 _min(INFINITY, INFINITY, INFINITY);
	Vec3 _max(-INFINITY, -INFINITY, -INFINITY);
	for(int i = 0; i < 2; i++)
		for(int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				double x = i * box.Max().x() + (1 - i) * box.Min().x();
				double y = j * box.Max().y() + (1 - j) * box.Min().y();
				double z = k * box.Max().z() + (1 - k) * box.Min().z();
				double newx = cos_theta * x + sin_theta * z;
				double newz = -sin_theta * x + cos_theta * z;
				Vec3 tester(newx, y, newz);
				for (int c = 0; c < 3; c++) {
					if (tester[c] > _max[c]) _max[c] = tester[c];
					if (tester[c] < _min[c]) _min[c] = tester[c];
				}
			}
	box = AABB(_min, _max);
}

bool RotateY::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	// 将光线的原点和方向旋转 -theta 度
	Vec3 origin = r.Origin().rotateY(-sin_theta, cos_theta);
	Vec3 direction = r.Direction().rotateY(-sin_theta, cos_theta);
	Ray rotatedRay(origin, direction, r.Time());
	if (object->hit(rotatedRay, t_min, t_max, info)) {
		info.position = info.position.rotateY(sin_theta, cos_theta);
		info.normal = info.normal.rotateY(sin_theta, cos_theta);
		return true;
	}
	return false;
}

AABB RotateY::GetBox() const {
	return box;
}
