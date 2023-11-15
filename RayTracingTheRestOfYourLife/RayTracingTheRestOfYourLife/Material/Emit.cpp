#include "Emit.h"

bool Emit::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
    // 光线到达光源之后, 就不再散射了
    return false;
}

Color Emit::emitted(double u, double v, const Point3& p) const {
    return emit->Value(u, v, p);
}
