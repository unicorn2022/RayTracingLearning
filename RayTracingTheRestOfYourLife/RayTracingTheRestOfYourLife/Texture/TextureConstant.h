#pragma once
#include "Texture.h"

class TextureConstant : public Texture {
public:
	/*
	* @param color 纹理颜色
	*/
	TextureConstant(const Color& color) : color(color) {}

	/*
	* @brief 获取纹理颜色
	* @param u 纹理坐标u
	* @param v 纹理坐标v
	* @param p 坐标
	* @return 纹理颜色
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Color color;
};