#include "ONB.h"

ONB::ONB(Vec3 normal) {
    build_from_w(normal);
}

Vec3 ONB::local(double x, double y, double z) const {
    return x * u() + y * v() + z * w();
}

Vec3 ONB::local(const Vec3& v) const {
    return local(v[0], v[1], v[2]);
}

void ONB::build_from_w(const Vec3& normal) {
    Vec3 n = normal.normalize();

    Vec3 vup(1, 0, 0);
    if (fabs(n.x()) > 0.9) vup = Vec3(0, 1, 0);

    axis[2] = n;
    axis[1] = n.cross(vup).normalize();
    axis[0] = n.cross(axis[1]).normalize();
}
