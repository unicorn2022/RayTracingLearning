#pragma once

#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {
public:
	vec3() { e[0] = e[1] = e[2] = 0; }
	vec3(double e0) { e[0] = e[1] = e[2] = e0; }
	vec3(double e0, double e1, double e2) { e[0] = e0, e[1] = e1, e[2] = e2; }

	// 取数组元素
	double x() const { return e[0]; }
	double y() const { return e[1]; }
	double z() const { return e[2]; }

	// 四则运算
	vec3 operator-() const;
	double operator[](int i) const;
	double& operator[](int i);
	vec3& operator+=(const vec3& v);
	vec3& operator*=(const double& t);
	vec3& operator/=(const double& t); 

	// 长度^2
	double length_squared() const;
	// 长度
	double length() const;
	// 单位化
	vec3 normalize() const;
	// gamma 2 矫正
	void gamma();

	/*
	* @brief 向量反射
	* @param normal	法线
	* @return 反射向量
	*/
	vec3 reflect(const vec3& normal) const;

	/*
	* @brief 向量折射
	* @param normal	法线
	* @param eta	折射率 η
	* @param r_out	折射向量
	* @return 是否发生折射
	*/
	bool refract(const vec3& normal, double eta, vec3& r_out) const;

	// 输出相关
	void write_color(std::ostream& out);

private:
	double e[3];
};

using point3 = vec3;
using color = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

inline vec3 operator*(double t, const vec3& u) {
	return vec3(u[0] * t, u[1] * t, u[2] * t);
}

inline vec3 operator*(const vec3& u, double t) {
	return t * u;
}

inline vec3 operator/(const vec3& u, double t) {
	return (1 / t) * u;
}

inline double dot(const vec3& u, const vec3& v) {
	return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(
		u[1] * v[2] - u[2] * v[1],
		u[2] * v[0] - u[0] * v[2],
		u[0] * v[1] - u[1] * v[0]
	);
}