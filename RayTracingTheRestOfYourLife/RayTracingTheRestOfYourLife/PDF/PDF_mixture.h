#pragma once
#include "PDF.h"
#include "../Utils/Utils.h"

class PDF_mixture : public PDF {
public:
	/*
	* @brief 等比例混合两个PDF
	* @param pdf1 第一个PDF
	* @param pdf2 第二个PDF
	* @param weight0 pdf1的权重
	*/
	PDF_mixture(Ref<PDF> pdf1, Ref<PDF> pdf2, double weight1);

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
	Ref<PDF> pdf[2];
	double weight1;
};

