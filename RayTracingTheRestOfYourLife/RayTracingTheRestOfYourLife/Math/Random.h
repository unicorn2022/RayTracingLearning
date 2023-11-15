#pragma once
#include <random>
#include <memory>
#include <vector>
#include "../Math/Vec3.h"

static const double PI = std::_Pi;

/* ����������� */
class Random {
public:
	/*
	* @brief ����һ��[0,1)�������
	*/
	static double rand01() {
		static std::mt19937 mt;
		static std::uniform_real_distribution<double> rand_double;
		return rand_double(mt);
	}

	/*
	* @brief ���ص�λ�����ϵ������
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