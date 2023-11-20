#pragma once

#include "../Math/Vec3.h"

class PDF {
public:
	/*
	* @brief 获取 direction 对应的概率密度
	* @param direction 随机方向
	* @return 概率密度
	*/
	virtual double value(const Vec3& direction) const = 0;

	/*
	* @brief 生成随机方向
	*/
	virtual Vec3 generate() const = 0;
};