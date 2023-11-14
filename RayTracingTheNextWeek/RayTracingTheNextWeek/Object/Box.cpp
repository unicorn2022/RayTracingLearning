#include "Box.h"
#include "RectXY.h"
#include "RectYZ.h"
#include "RectXZ.h"
#include "Transform/FlipNormal.h"

Box::Box(const Vec3& point_min, const Vec3& point_max, Ref<Material> material) 
	:point_min(point_min), point_max(point_max) {
	sides = New<ObjectWorld>();
	sides->Add(New<RectXY>(point_min.x(), point_max.x(), point_min.y(), point_max.y(), point_max.z(), material));
	sides->Add(New<FlipNormal>(New<RectXY>(point_min.x(), point_max.x(), point_min.y(), point_max.y(), point_min.z(), material)));
	sides->Add(New<RectXZ>(point_min.x(), point_max.x(), point_min.z(), point_max.z(), point_max.y(), material));
	sides->Add(New<FlipNormal>(New<RectXZ>(point_min.x(), point_max.x(), point_min.z(), point_max.z(), point_min.y(), material)));
	sides->Add(New<RectYZ>(point_min.y(), point_max.y(), point_min.z(), point_max.z(), point_max.x(), material));
	sides->Add(New<FlipNormal>(New<RectYZ>(point_min.y(), point_max.y(), point_min.z(), point_max.z(), point_min.x(), material)));
}

bool Box::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	return sides->hit(r, t_min, t_max, info);
}

AABB Box::GetBox() const {
	return AABB(point_min, point_max);
}
