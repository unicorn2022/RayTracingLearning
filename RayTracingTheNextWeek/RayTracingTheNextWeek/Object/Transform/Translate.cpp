#include "Translate.h"

bool Translate::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    Ray moved_r(r.Origin() - offset, r.Direction(), r.Time());
    if (object->hit(moved_r, t_min, t_max, info)) {
        info.position += offset;
        return true;
    }
    return false;
}

AABB Translate::GetBox() const {
    auto box = object->GetBox();
    return AABB(box.Min() + offset, box.Max() + offset);
}
