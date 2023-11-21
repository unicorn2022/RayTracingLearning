#include "RectXZ.h"
#include "../Math/Random.h"

bool RectXZ::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
    double t = (k - r.Origin().y()) / r.Direction().y();
    if (t < t_min || t > t_max) return false;

    Point3 p = r.At(t);
    if (p.x() < x1 || p.x() > x2) return false;
    if (p.z() < z1 || p.z() > z2) return false;

    info.t = t;
    info.position = p;
    info.normal = Vec3(0, 1, 0);
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

double RectXZ::pdf_value(const Point3& origin, const Vec3 direction) const {
    HitInfo info;
    if (this->hit(Ray(origin, direction), 1e-3, INFINITY, info)) {
        double area = (x2 - x1) * (z2 - z1);
        double distance_squared = info.t * info.t * direction.length_squared();
        double cosine = fabs(direction.dot(info.normal) / direction.length());
        // pdf = distance^2 / (cos_alpha * A)
        return distance_squared / (cosine * area);
    }
    else return 0;
}

Vec3 RectXZ::random(const Point3& origin) const {
    Point3 on_light = Vec3(Random::rand_between(x1, x2), k, Random::rand_between(z1, z2));
    return on_light - origin;
}
