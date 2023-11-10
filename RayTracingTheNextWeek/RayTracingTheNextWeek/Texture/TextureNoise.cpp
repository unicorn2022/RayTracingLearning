#include "TextureNoise.h"

Color TextureNoise::Value(double u, double v, const Point3& p) const {
    return Vec3(1, 1, 1) * noise.noise(p);
}
