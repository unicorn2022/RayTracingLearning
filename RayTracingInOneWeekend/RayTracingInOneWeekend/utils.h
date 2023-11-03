#pragma once

#include <random>
#include "vec3.h"

/* 随机数生成器 */
class Random {
public:
	/*
	* @brief 返回一个[0,1)的随机数
	*/
	static double random_double_01() { 
		static std::mt19937 mt;
		static std::uniform_real_distribution<double> rand_double;
		return rand_double(mt);
	}
	
	/*
	* @brief 返回单位球中的随机点
	*/
	static vec3 random_unit_sphere() {
		vec3 p;
		do {
			p = 2 * vec3(random_double_01(), random_double_01(), random_double_01()) - vec3(1.0f);
		} while (p.length_squared() >= 1.0);
		return p;
	}
};



