#include "Emit.h"

bool Emit::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
    // ���ߵ����Դ֮��, �Ͳ���ɢ����
    return false;
}

Color Emit::emitted(double u, double v, const Point3& p) const {
    return emit->Value(u, v, p);
}
