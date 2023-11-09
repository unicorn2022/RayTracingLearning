#pragma once

#include "ObjectBase.h"
#include "BVHnode.h"
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

	/*
	* @brief 获取当前对象的包围盒
	*/
	AABB GetBox() const { return AABB(); }

	/*
	* @brief 获取场景中的物体
	* @param index 物体的索引
	*/
	Ref<ObjectBase> GetObject(int index) const { return objects[index]; }
	
	/*
	* @brief 为当前场景构建 BVH
	*/
	void Build();

private:
	
	/*
	* @brief 构建 BVH, 按照深度, 分别将当前深度的节点按照XYZ轴进行排序
	* @param u 当前节点
	* @param L 当前节点控制的区间的左端点
	* @param R 当前节点控制的区间的右端点
	* @param deep 当前递归深度
	*/
	void build(Ref<BVHnode> u, int L, int R, int deep = 0);

	std::vector<Ref<ObjectBase>> objects;
	Color background;
	Ref<BVHnode> root;

};

