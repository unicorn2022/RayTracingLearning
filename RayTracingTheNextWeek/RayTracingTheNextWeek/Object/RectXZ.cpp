#include "RectXZ.h"

bool RectXZ::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    double t = (k - r.Origin().y()) / r.Direction().y();
    if (t < t_min || t > t_max) return false;

    Point3 p = r.At(t);
    if (p.x() < x1 || p.x() > x2) return false;
    if (p.z() < z1 || p.z() > z2) return false;

    info.t = t;
    info.position = p;
    info.set_face_normal(r, Vec3(0, 1, 0));
    info.material = material;
    info.u = (p.x() - x1) / (x2 - x1);
    info.v = (p.z() - z1) / (z2 - z1);
    return true;
}

AABB RectXZ::GetBox() const {
    return AABB(
        Vec3(x1, k - 0.0001, z1),
        Vec3(x2, k + 0.0001, z2)
    );
}
