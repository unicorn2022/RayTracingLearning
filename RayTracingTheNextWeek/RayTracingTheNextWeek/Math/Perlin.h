#pragma once
#include "Vec3.h"

class Perlin {
public:
	/*
	* @brief ��ȡλ��p��������ֵ
	* @param p λ��
	* @return ����ֵ
	*/
	double noise(const Point3& p)const;

	/*
	* @brief ��ȡλ��p���Ķ��Ƶ����ϵĸ�������ֵ
	* @param p λ��
	* @param depth Ƶ��������
	* @return ����ֵ
	*/
	double turb(const Point3& p, int depth = 7) const;

private:
	/*
	* @brief �����Բ�ֵ
	*/
	double trilinear_interp(Point3 list[2][2][2], double u, double v, double w)const;

public:
	static Point3* perlin_generate();
	static int* perlin_generate_perm();

private:
	static Point3* random_value;	// ���������, ÿ��Ԫ����[0, 1)֮��������
	static int* perm_x;	// �����������, Ϊ 0~255 ���������
	static int* perm_y;	// �����������, Ϊ 0~255 ���������
	static int* perm_z;	// �����������, Ϊ 0~255 ���������
};

