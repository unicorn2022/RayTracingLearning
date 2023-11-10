#pragma once
#include "Vec3.h"

class Perlin {
public:
	/*
	* @brief 获取位置p处的噪声值
	* @param p 位置
	* @return 噪声值
	*/
	double noise(const Point3& p)const;

	/*
	* @brief 获取位置p处的多个频率组合的复合噪声值
	* @param p 位置
	* @param depth 频率种类数
	* @return 噪声值
	*/
	double turb(const Point3& p, int depth = 7) const;

private:
	/*
	* @brief 三线性插值
	*/
	double trilinear_interp(Point3 list[2][2][2], double u, double v, double w)const;

public:
	static Point3* perlin_generate();
	static int* perlin_generate_perm();

private:
	static Point3* random_value;	// 总随机序列, 每个元素是[0, 1)之间的随机数
	static int* perm_x;	// 分量随机序列, 为 0~255 的随机排列
	static int* perm_y;	// 分量随机序列, 为 0~255 的随机排列
	static int* perm_z;	// 分量随机序列, 为 0~255 的随机排列
};

