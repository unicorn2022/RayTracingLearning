#pragma once
#include "PDF.h"
#include "../Math/ONB.h"

class PDF_cos : public PDF {
public:
	/*
	* @brief 随机方向满足 cos 分布
	* @param normal 法线
	*/
	PDF_cos(const Vec3& normal) : uvw(normal) {}

	/*
	* @brief 获取 direction 对应的概率密度
	* @param direction 随机方向
	* @return 概率密度
	*/
	virtual double value(const Vec3& direction) const override;

	/*
	* @brief 生成随机方向
	*/
	virtual Vec3 generate() const override;

private:
	ONB uvw;
};

