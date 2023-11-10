#include "TextureChecker.h"

Color TextureChecker::Value(double u, double v, const Point3& p) const {
    double sines = sin(30 * p.x()) * sin(30 * p.y()) * sin(30 * p.z());
    if (sines < 0) return odd->Value(u, v, p);
    else return even->Value(u, v, p);
}
