#include "Box.h"

bool Box::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	return false;
}

AABB Box::GetBox() const {
	return AABB();
}
