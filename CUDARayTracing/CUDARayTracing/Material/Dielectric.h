#pragma once
#include "MaterialBase.h"
class Dielectric : public MaterialBase {
public:
	/*
	* @param refractive_index 折射率
	*/
	__host__ __device__ Dielectric(double refractive_index) : refractive_index(refractive_index) {}

	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param info 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	__host__ __device__  bool scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const override;

private:
	/*
	* @brief 计算反射系数
	* @param cosine 入射角余弦值
	* @param refractive_index 折射率
	*/
	__host__ __device__ double schlick(const double cosine, const double refractive_index) const;


	double refractive_index; // 折射率
};
