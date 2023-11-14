#include "ConstantMedium.h"
#include "../../Math/Random.h"

bool ConstantMedium::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    HitInfo info1, info2;
    if (object->hit(r, -INFINITY, INFINITY, info1) && object->hit(r, info1.t + 0.0001, INFINITY, info2)) {
        if (info1.t < t_min) info1.t = t_min;
        if (info2.t > t_max) info2.t = t_max;
        if (info1.t >= info2.t) return false;
        
        if (info1.t < 0) info1.t = 0;
        float distance_inside_boundary = (info2.t - info1.t) * r.Direction().length();
        float hit_distance = -(1 / density) * log(Random::rand01());
        if (hit_distance < distance_inside_boundary) {
            info.t = info1.t + hit_distance / r.Direction().length();
            info.position = r.At(info.t);
            info.normal = Vec3(Random::rand01(), Random::rand01(), Random::rand01());
            info.material = material;
            return true;
        }
    }
    return false;
}

AABB ConstantMedium::GetBox() const {
    return object->GetBox();
}
