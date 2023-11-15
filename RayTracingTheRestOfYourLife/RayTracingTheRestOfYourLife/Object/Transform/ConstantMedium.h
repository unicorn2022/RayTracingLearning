#pragma once
#include "../Object.h"
#include "../../Material/Isotropic.h"
#include "../../Texture/Texture.h"

class ConstantMedium : public Object {
public:
	/*
	* @brief 恒定密度介质
	* @param object 介质的形状
	* @param density 介质的密度
	* @param albedo 介质的反射率
	*/
	ConstantMedium(Ref<Object> object, double density, Ref<Texture> albedo) : object(object), density(density), isotropic_material(New<Isotropic>(albedo)) {}

	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param info 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const Ray& r, double t_min, double t_max, HitInfo& info) const override;

	/*
	* @brief 获取当前对象的包围盒
	*/
	virtual AABB GetBox() const override;
private:
	Ref<Object> object;
	double density;
	Ref<Isotropic> isotropic_material;
};

