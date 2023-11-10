#pragma once
#include "Texture.h"
#include "../Math/Perlin.h"

class TextureNoise : public Texture {
public:
	TextureNoise() {}

	/*
	* @brief 获取纹理颜色
	* @param u 纹理坐标u
	* @param v 纹理坐标v
	* @param p 坐标
	* @return 纹理颜色
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Perlin noise;
};

