#pragma once
#include "ray.h"
#include "hittable.h"
#include "utils.h"

struct hit_record;

class material {
public:
	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param record 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	virtual bool scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const = 0;
};

class material_diffuse : public material {
public:
	/*
	* @param albedo 反射率
	*/
	material_diffuse(const color& albedo) : albedo(albedo) {}
	
	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param record 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	virtual bool scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const;

private:
	color albedo;
};

class material_metal : public material {
public:
	/*
	* @param albedo 反射率
	* @param fuzz 模糊度, 0为完全反射, 1为完全散射
	*/
	material_metal(const color& albedo, double fuzz = 0) : albedo(albedo), fuzz(fuzz) {}

	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param record 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	virtual bool scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const;

private:
	color albedo;

	double fuzz;
};

class material_dielectric : public material {
public:
	/*
	* @param refractive_index 折射率
	*/
	material_dielectric(double refractive_index) : refractive_index(refractive_index) {}

	/*
	* @brief 生成散射光线
	* @param r_in 入射光线
	* @param record 碰撞信息
	* @param attenuation 当发生散射时, 光强如何衰减, 分为rgb三个分量
	* @param r_out 散射光线
	* @return 是否得到了散射光线
	*/
	virtual bool scatter(const ray& r_in, const hit_record& record, color& attenuation, ray& r_out) const;

	/*
	* @brief 计算反射系数
	* @param cosine 入射角余弦值
	* @param refractive_index 折射率 
	*/
	inline double schlick(const double cosine, const double refractive_index) const;

private:
	double refractive_index; // 折射率
};