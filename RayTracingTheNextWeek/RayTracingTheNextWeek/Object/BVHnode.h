#pragma once
#include "Object.h"
#include <vector>

class BVHnode : public Object {
public:
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

	void Update();

public:
	Ref<BVHnode> left;
	Ref<BVHnode> right;
	Ref<Object> object;
	int L, R;
	AABB box;
};

