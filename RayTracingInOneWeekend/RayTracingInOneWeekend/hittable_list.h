#pragma once

#include "hittable.h"
#include "vec3.h"
#include "utils.h"
#include <memory>
#include <vector>


class hittable_list : public hittable {
public:
	hittable_list(color background) : background(background){}

	void clear() { objects.clear(); }
	void add(std::shared_ptr<hittable> object) {
		objects.push_back(object);
	}

	/*
	* @brief 判断光线是否与当前对象碰撞
	* @param r 光线
	* @param t_min 光线的最小 t 值
	* @param t_max 光线的最大 t 值
	* @param record 碰撞点信息
	* @return 是否碰撞
	*/
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& record) const override;

	/*
	* @brief 计算当前得到的颜色 
	* @param r 光线
	* @param depth 递归深度
	* @return 当前得到的颜色
	*/
	color ray_color(const ray& r, int depth = 0);

private:
	std::vector<std::shared_ptr<hittable>> objects;
	color background;
};