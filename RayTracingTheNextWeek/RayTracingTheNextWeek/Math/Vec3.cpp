#include "Vec3.h"

Vec3 Vec3::operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
}

double Vec3::operator[](int i) const {
    return e[i];
}

double& Vec3::operator[](int i) {
    return e[i];
}

Vec3& Vec3::operator+=(const Vec3& v) {
    this->e[0] += v.e[0];
    this->e[1] += v.e[1];
    this->e[2] += v.e[2];
    return *this;
}

Vec3& Vec3::operator*=(const double& t) {
    this->e[0] *= t;
    this->e[1] *= t;
    this->e[2] *= t;
    return *this;
}

Vec3& Vec3::operator/=(const double& t) {
    return *this *= (1 / t);
}

Vec3 Vec3::operator+(const Vec3& v) const {
    return Vec3(e[0] + v[0], e[1] + v[1], e[2] + v[2]);
}

Vec3 Vec3::operator-(const Vec3& v) const {
    return Vec3(e[0] - v[0], e[1] - v[1], e[2] - v[2]);
}

Vec3 Vec3::operator*(const Vec3& v) const {
    return Vec3(e[0] * v[0], e[1] * v[1], e[2] * v[2]);
}

Vec3 Vec3::operator*(const double& t) const {
    return Vec3(e[0] * t, e[1] * t, e[2] * t);
}

Vec3 Vec3::operator/(const double& t) const {
    return (*this) * (1 / t);
}

bool Vec3::operator<(const Vec3& v) const {
    for(int i = 0; i < 3; i++)
        if(e[i] != v[i]) return e[i] < v[i];
    return true;
}

bool Vec3::operator==(const Vec3& v) const
{
    for (int i = 0; i < 3; i++)
        if (e[i] - v[i] > 1e-5) return false;
    return true;
}

bool Vec3::operator!=(const Vec3& v) const {
    return !(*this == v);
}

double Vec3::dot(const Vec3& v) const {
    return e[0] * v[0] + e[1] * v[1] + e[2] * v[2];
}

Vec3 Vec3::cross(const Vec3& v) const {
    return Vec3(
        e[1] * v[2] - e[2] * v[1],
		e[2] * v[0] - e[0] * v[2],
		e[0] * v[1] - e[1] * v[0]
    );
}

double Vec3::length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

double Vec3::length() const {
    return std::sqrt(length_squared());
}

Vec3 Vec3::normalize() const {
    double len = this->length();
    return Vec3(e[0] / len, e[1] / len, e[2] / len);
}

void Vec3::gamma() {
    e[0] = pow(e[0], 1.0 / 2.2);
    e[1] = pow(e[1], 1.0 / 2.2);
    e[2] = pow(e[2], 1.0 / 2.2);
}

Vec3 Vec3::reflect(const Vec3& normal) const {
    return *this - 2 * this->dot(normal) * normal;
}

bool Vec3::refract(const Vec3& normal, double eta, Vec3& r_out) const {
    Vec3 unit_in = this->normalize();

    double cos1 = -unit_in.dot(normal);
    double cos2 = 1 - eta * eta * (1 - cos1 * cos1);

    // θ2 <= 90°, 说明没有发生全反射
    if (cos2 > 0) {
        r_out = eta * (*this) + normal * (eta * cos1 - sqrt(cos2));
        return true;
    }
    // 发生全反射
    return false;
}

void Vec3::write_color(std::ostream& out) {
    out << static_cast<int>(255.99 * e[0]) << " "
        << static_cast<int>(255.99 * e[1]) << " "
        << static_cast<int>(255.99 * e[2]) << "\n";
}
