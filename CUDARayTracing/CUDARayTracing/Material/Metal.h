#pragma once
#include "MaterialBase.h"
class Metal : public MaterialBase {
public:
	/*
	* @param albedo 反射率
	* @param fuzz 模糊度, 0为完全反射, 1为完全散射
	*/
	__host__ __device__ Metal(const Color& albedo, double fuzz = 0) : albedo(albedo), fuzz(fuzz) {}


	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param info 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	__host__ __device__ virtual bool scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const override;

private:
	Color albedo;
	double fuzz;
};

