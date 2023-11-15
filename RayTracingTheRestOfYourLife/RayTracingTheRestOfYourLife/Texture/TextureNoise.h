#pragma once
#include "Texture.h"
#include "../Math/Perlin.h"

class TextureNoise : public Texture {
public:
	/*
	* @param scale ��������
	*/
	TextureNoise(double scale = 1.0) : scale(scale) {}

	/*
	* @brief ��ȡ������ɫ
	* @param u ��������u
	* @param v ��������v
	* @param p ����
	* @return ������ɫ
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Perlin noise;
	double scale;
};

