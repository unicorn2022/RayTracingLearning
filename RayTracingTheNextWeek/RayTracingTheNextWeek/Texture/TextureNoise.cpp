#include "TextureNoise.h"
#include <cmath>

Color TextureNoise::Value(double u, double v, const Point3& p) const {
    return Vec3(1, 1, 1) * 0.5 * (1 + sin(scale * p.z() + 10 * noise.turb(p)));
}
