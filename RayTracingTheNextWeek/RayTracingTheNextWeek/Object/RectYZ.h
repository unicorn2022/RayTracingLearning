#pragma once
#include "Object.h"
class RectYZ : public Object {
public:
	/*
	* @param y1 y 轴最小值
	* @param y2 y 轴最大值
	* @param z1 z 轴最小值
	* @param z2 z 轴最大值
	* @param k	x 轴值
	* @param material 材质
	*/
	RectYZ(double y1, double y2, double z1, double z2, double k, Ref<Material> material)
		: z1(z1), z2(z2), y1(y1), y2(y2), k(k), material(material) {}

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
	Ref<Material> material;
	double y1, y2, z1, z2, k;
};

