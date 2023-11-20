#pragma once
#include "Object.h"
class RectXZ : public Object {
public:
	/*
	* @param x1 x 轴最小值
	* @param x2 x 轴最大值
	* @param z1 z 轴最小值
	* @param z2 z 轴最大值
	* @param k	y 轴值
	* @param material 材质
	*/
	RectXZ(double x1, double x2, double z1, double z2, double k, Ref<Material> material)
		: x1(x1), x2(x2), z1(z1), z2(z2), k(k), material(material) {}

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

	/*
	* @brief 获取采样方向的 pdf 值
	* @param origin 观察点
	* @param direction 观察方向
	*/
	virtual double pdf_value(const Point3& origin, const Vec3 direction) const override;

	/*
	* @brief 在当前对象的表面随机采样一个点, 从 origin 看向该点
	* @param origin 观察点
	* @return 观察方向
	*/
	virtual Vec3 random(const Point3& origin) const override;

private:
	Ref<Material> material;
	double x1, x2, z1, z2, k;
};

