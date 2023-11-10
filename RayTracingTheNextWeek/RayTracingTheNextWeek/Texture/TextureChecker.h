#pragma once
#include "Texture.h"

class TextureChecker : public Texture {
public:
	/*
	* @param odd 奇数纹理
	* @param even 偶数纹理
	*/
	TextureChecker(Ref<Texture> odd, Ref<Texture> even) : odd(odd), even(even) {}

	/*
	* @brief 获取纹理颜色
	* @param u 纹理坐标u
	* @param v 纹理坐标v
	* @param p 坐标
	* @return 纹理颜色
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Ref<Texture> odd;
	Ref<Texture> even;
};

