#include "Emit.h"

bool Emit::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
    // 光线到达光源之后, 就不再散射了
    return false;
}

Color Emit::emitted(const Ray& r_in, const HitInfo& info, double u, double v, const Point3& p) const {
    if (info.normal.dot(r_in.Direction()) < 0)
        return emit->Value(u, v, p);
    else
        return Color();
}
