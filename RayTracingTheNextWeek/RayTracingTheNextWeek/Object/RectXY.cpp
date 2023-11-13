#include "RectXY.h"

bool RectXY::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    double t = (k - r.Origin().z()) / r.Direction().z();
    if (t < t_min || t > t_max) return false;

    Point3 p = r.At(t);
    if (p.x() < x1 || p.x() > x2) return false;
    if (p.y() < y1 || p.y() > y2) return false;

    info.t = t;
    info.position = p;
    info.set_face_normal(r, Vec3(0, 0, 1));
    info.material = material;
    info.u = (p.x() - x1) / (x2 - x1);
    info.v = (p.y() - y1) / (y2 - y1);
    return true;
}

AABB RectXY::GetBox() const {
    return AABB(
        Vec3(x1, y1, k - 0.0001),
        Vec3(x2, y2, k + 0.0001)
    );
}
