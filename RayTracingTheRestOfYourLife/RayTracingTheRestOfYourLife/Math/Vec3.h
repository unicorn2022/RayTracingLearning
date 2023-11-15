#pragma once

#include <cmath>
#include <iostream>

class Vec3 {
public:
	Vec3() { e[0] = e[1] = e[2] = 0; }
	Vec3(double e0) { e[0] = e[1] = e[2] = e0; }
	Vec3(double e0, double e1, double e2) { e[0] = e0, e[1] = e1, e[2] = e2; }

	// 取数组元素
	double x() const { return e[0]; }
	double y() const { return e[1]; }
	double z() const { return e[2]; }

public:
	Vec3 operator-() const;
	double operator[](int i) const;
	double& operator[](int i);
	Vec3& operator+=(const Vec3& v);
	Vec3& operator*=(const double& t);
	Vec3& operator/=(const double& t);
	Vec3 operator+(const Vec3& v) const;
	Vec3 operator-(const Vec3& v) const;
	Vec3 operator*(const Vec3& v) const;
	Vec3 operator*(const double& t) const;
	Vec3 operator/(const double& t) const;
	bool operator<(const Vec3& v) const;
	bool operator==(const Vec3& v) const;
	bool operator!=(const Vec3& v) const;
	double dot(const Vec3& v) const;
	Vec3 cross(const Vec3& v) const;


public:
	// 长度^2
	double length_squared() const;
	// 长度
	double length() const;
	// 单位化
	Vec3 normalize() const;
	// gamma 矫正
	void gamma();

	/*
	* @brief 向量反射
	* @param normal	法线
	* @return 反射向量
	*/
	Vec3 reflect(const Vec3& normal) const;

	/*
	* @brief 向量折射
	* @param normal	法线
	* @param eta	折射率 η
	* @param r_out	折射向量
	* @return 是否发生折射
	*/
	bool refract(const Vec3& normal, double eta, Vec3& r_out) const;

	/*
	* @brief 向量绕Y轴旋转 theta
	* @param sin_theta	正弦值
	* @param cos_theta	余弦值
	* @return 旋转后的向量
	*/
	Vec3 rotateY(double sin_theta, double cos_theta) const;

	// 输出相关
	void write_color(std::ostream& out) const;
	std::string Print() const;

private:
	double e[3];
};

using Point3 = Vec3;
using Color = Vec3;

inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
	return out << v[0] << ' ' << v[1] << ' ' << v[2];
}
inline Vec3 operator*(double t, const Vec3& u) {
	return Vec3(u[0] * t, u[1] * t, u[2] * t);
}