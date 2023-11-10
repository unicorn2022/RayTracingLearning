#pragma once
#include "Vec3.h"

class Perlin {
public:
	double noise(const Vec3& p)const;

public:
	static double* perlin_generate();

	static int* perlin_generate_perm();

	static void permute(int p[], int n);


private:
	static double* random_value;	// 总随机序列, 每个元素是[0, 1)之间的随机数
	static int* perm_x;	// 分量随机序列, 为 0~255 的随机排列
	static int* perm_y;	// 分量随机序列, 为 0~255 的随机排列
	static int* perm_z;	// 分量随机序列, 为 0~255 的随机排列
};

