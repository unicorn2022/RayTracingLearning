#pragma once
#include "Texture.h"

class TextureConstant : public Texture {
public:
	/*
	* @param color ������ɫ
	*/
	TextureConstant(const Color& color) : color(color) {}

	/*
	* @brief ��ȡ������ɫ
	* @param u ��������u
	* @param v ��������v
	* @param p ����
	* @return ������ɫ
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Color color;
};