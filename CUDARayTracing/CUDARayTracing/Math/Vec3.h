#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <iostream>

class Vec3 {
public:
	__host__ __device__ Vec3() { e[0] = e[1] = e[2] = 0; }
	__host__ __device__ Vec3(double e0) { e[0] = e[1] = e[2] = e0; }
	__host__ __device__ Vec3(double e0, double e1, double e2) { e[0] = e0, e[1] = e1, e[2] = e2; }

	// 取数组元素
	__host__ __device__ double x() const { return e[0]; }
	__host__ __device__ double y() const { return e[1]; }
	__host__ __device__ double z() const { return e[2]; }

public:
	__host__ __device__ Vec3 operator-() const;
	__host__ __device__ double operator[](int i) const;
	__host__ __device__ double& operator[](int i);
	__host__ __device__ Vec3& operator+=(const Vec3& v);
	__host__ __device__ Vec3& operator*=(const double& t);
	__host__ __device__ Vec3& operator/=(const double& t);
	__host__ __device__ Vec3 operator+(const Vec3& v) const;
	__host__ __device__ Vec3 operator-(const Vec3& v) const;
	__host__ __device__ Vec3 operator*(const Vec3& v) const;
	__host__ __device__ Vec3 operator*(const double& t) const;
	__host__ __device__ Vec3 operator/(const double& t) const;
	__host__ __device__ double dot(const Vec3& v) const;
	__host__ __device__ Vec3 cross(const Vec3& v) const;


public:
	// 长度^2
	__host__ __device__ double length_squared() const;
	// 长度
	__host__ __device__ double length() const;
	// 单位化
	__host__ __device__ Vec3 normalize() const;
	// gamma 矫正
	__host__ __device__ void gamma();

	/*
	* @brief 向量反射
	* @param normal	法线
	* @return 反射向量
	*/
	__host__ __device__ Vec3 reflect(const Vec3& normal) const;

	/*
	* @brief 向量折射
	* @param normal	法线
	* @param eta	折射率 η
	* @param r_out	折射向量
	* @return 是否发生折射
	*/
	__host__ __device__ bool refract(const Vec3& normal, double eta, Vec3& r_out) const;

	// 输出相关
	__host__ __device__ void write_color(std::ostream& out);

private:
	double e[3];
};

using Point3 = Vec3;
using Color = Vec3;

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
	return out << v[0] << ' ' << v[1] << ' ' << v[2];
}
__host__ __device__ inline Vec3 operator*(double t, const Vec3& u) {
	return Vec3(u[0] * t, u[1] * t, u[2] * t);
}