#pragma once
#include <random>
#include <memory>
#include <vector>
#include "../Math/Vec3.h"

static const double PI = 3.1415926535;

/* 随机数生成器 */
class Random {
public:
	/*
	* @brief 返回一个[0,1)的随机数
	*/
	static double rand01() {
		static std::mt19937 mt;
		static std::uniform_real_distribution<double> rand_double;
		return rand_double(mt);
	}

	/*
	* @brief 返回一个[L,R)的随机数
	*/
	static double rand_between(double L, double R) {
		return L + rand01() * (R - L);
	}

	/*
	* @brief 返回单位球面上的随机点
	*/
	static Vec3 rand_unit_sphere() {
		double theta = rand01() * 2 * PI;
		//double phi = rand01() * PI;
		double phi = acos(1 - 2 * rand01());

		double x = sin(phi) * cos(theta);
		double y = sin(phi) * sin(theta);
		double z = cos(phi);

		return Vec3(x, y, z);
	}

	/*
	* @brief 返回随即方向, 满足 pdf(direction) = cos θ / π
	*/
	static Vec3 rand_cosine_direction() {
		double r1 = rand01();
		double r2 = rand01();

		double x = cos(2 * PI * r1) * sqrt(r2);
		double y = sin(2 * PI * r1) * sqrt(r2);
		double z = sqrt(1 - r2);
		return Vec3(x, y, z);
	}
};