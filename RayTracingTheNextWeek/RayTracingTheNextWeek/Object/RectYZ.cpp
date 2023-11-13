#include "RectYZ.h"

bool RectYZ::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    double t = (k - r.Origin().x()) / r.Direction().x();
    if (t < t_min || t > t_max) return false;

    Point3 p = r.At(t);
    if (p.y() < y1 || p.y() > y2) return false;
    if (p.z() < z1 || p.z() > z2) return false;

    info.t = t;
    info.position = p;
    info.set_face_normal(r, Vec3(1, 0, 0));
    info.material = material;
    info.u = (p.y() - y1) / (y2 - y1);
    info.v = (p.z() - z1) / (z2 - z1);
    return true;
}

AABB RectYZ::GetBox() const {
    return AABB(
        Vec3(k - 0.0001, y1, z1),
        Vec3(k + 0.0001, y2, z2)
    );
}
