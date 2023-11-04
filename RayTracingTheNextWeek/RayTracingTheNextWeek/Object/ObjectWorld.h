#pragma once

#include "ObjectBase.h"
#include "../Math/Vec3.h"
#include "../Utils/Utils.h"

class ObjectWorld : public ObjectBase {
public:
	/*
	* @param background: 背景颜色
	*/
	ObjectWorld(Color background) : background(background) {}

	void Clear() { objects.clear(); }
	void Add(Ref<ObjectBase> object) { objects.push_back(object); }
	
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
	* @brief 计算当前光线得到的颜色
	* @param r 光线
	* @param depth 递归深度
	* @return 当前得到的颜色
	*/
	Color GetColor(const Ray& r, int depth = 0);

private:
	std::vector<Ref<ObjectBase>> objects;
	Color background;
};

