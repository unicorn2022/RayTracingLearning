#pragma once
#include "PDF.h"
#include "../Object/Object.h"

class PDF_hit : public PDF {
public:
	/*
	* @brief 直接光源采样
	* @param light 光源
	* @param origin 观察点
	*/
	PDF_hit(Ref<Object> light, Point3 origin) : light(light), origin(origin) {}

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
	Point3 origin;
	Ref<Object> light;
};

