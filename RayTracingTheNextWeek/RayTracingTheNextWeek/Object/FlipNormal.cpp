#include "FlipNormal.h"

bool FlipNormal::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	if(object->hit(r, t_min, t_max, info)) {
		info.normal = -info.normal;
		return true;
	}
	else {
		return false;
	}
}

AABB FlipNormal::GetBox() const {
	return object->GetBox();
}
