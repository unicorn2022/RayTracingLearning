#pragma once
#include "Texture.h"
#include "../Math/Perlin.h"

class TextureNoise : public Texture {
public:
	TextureNoise() {}

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
};

