#pragma once
#include "../Math/Vec3.h"

class Texture {
public:
	/*
	* @brief ��ȡ������ɫ
	* @param u ��������u
	* @param v ��������v
	* @param p ����
	* @return ������ɫ
	*/
	virtual Color Value(double u, double v, const Point3& p) const = 0;
};

