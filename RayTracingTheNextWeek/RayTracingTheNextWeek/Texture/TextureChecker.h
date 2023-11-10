#pragma once
#include "Texture.h"

class TextureChecker : public Texture {
public:
	/*
	* @param odd ��������
	* @param even ż������
	*/
	TextureChecker(Ref<Texture> odd, Ref<Texture> even) : odd(odd), even(even) {}

	/*
	* @brief ��ȡ������ɫ
	* @param u ��������u
	* @param v ��������v
	* @param p ����
	* @return ������ɫ
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Ref<Texture> odd;
	Ref<Texture> even;
};

