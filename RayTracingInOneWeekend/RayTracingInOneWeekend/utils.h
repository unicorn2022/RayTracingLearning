#pragma once

#include <random>
#include "vec3.h"

/* ����������� */
class Random {
public:
	/*
	* @brief ����һ��[0,1)�������
	*/
	static double random_double_01() { 
		static std::mt19937 mt;
		static std::uniform_real_distribution<double> rand_double;
		return rand_double(mt);
	}
	
	/*
	* @brief ���ص�λ���е������
	*/
	static vec3 random_unit_sphere() {
		vec3 p;
		do {
			p = 2 * vec3(random_double_01(), random_double_01(), random_double_01()) - vec3(1.0f);
		} while (p.length_squared() >= 1.0);
		return p;
	}
};



