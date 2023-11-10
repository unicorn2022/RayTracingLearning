#pragma once
#include <random>
#include <memory>
#include <vector>
#include "../Math/Vec3.h"

static const double PI = std::_Pi;

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
	* @brief 返回单位球面上的随机点
	*/
	static Vec3 rand_unit_sphere() {
		double theta = rand01() * 2 * PI;
		double phi = rand01() * PI;

		double x = sin(phi) * cos(theta);
		double y = sin(phi) * sin(theta);
		double z = cos(phi);

		return Vec3(x, y, z);
	}
};