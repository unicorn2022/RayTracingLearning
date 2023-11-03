#include "vec3.h"

vec3 vec3::operator-() const {
	return vec3(-e[0], -e[1], -e[2]);
}

double vec3::operator[](int i) const {
	return e[i];
}

double& vec3::operator[](int i) {
	return e[i];
}

vec3& vec3::operator+=(const vec3& v) {
	this->e[0] += v.e[0];
	this->e[1] += v.e[1];
	this->e[2] += v.e[2];
	return *this;
}

vec3& vec3::operator*=(const double& t) {
	this->e[0] *= t;
	this->e[1] *= t;
	this->e[2] *= t;
	return *this;
}

vec3& vec3::operator/=(const double& t) {
	return *this *= (1 / t);
}

double vec3::length_squared() const {
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

double vec3::length() const {
	return sqrt(length_squared());
}

vec3 vec3::normalize() const {
	double len = this->length();
	return vec3(e[0] / len, e[1] / len, e[2] / len);
}

void vec3::gamma() {
	for (int i = 0; i < 3; i++)
		e[i] = pow(e[i], 1.0 / 2.2);
}

vec3 vec3::reflect(const vec3& normal) const {
	return *this - 2 * dot(*this, normal) * normal;
}

bool vec3::refract(const vec3& normal, double eta, vec3& r_out) const {
	vec3 unit_in = this->normalize();

	double cos1 = dot(-unit_in, normal);
	double cos2 = 1 - eta * eta * (1 - cos1 * cos1);

	// θ2 <= 90°, 说明没有发生全反射
	if (cos2 > 0) {
		r_out = eta * (*this) + normal * (eta * cos1 - sqrt(cos2));
		return true;
	}
	// 发生全反射
	return false;
}

void vec3::write_color(std::ostream& out) {
	out << static_cast<int>(255.99 * e[0]) << " "
		<< static_cast<int>(255.99 * e[1]) << " "
		<< static_cast<int>(255.99 * e[2]) << "\n";
}
