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
	static double* random_value;	// ���������, ÿ��Ԫ����[0, 1)֮��������
	static int* perm_x;	// �����������, Ϊ 0~255 ���������
	static int* perm_y;	// �����������, Ϊ 0~255 ���������
	static int* perm_z;	// �����������, Ϊ 0~255 ���������
};

