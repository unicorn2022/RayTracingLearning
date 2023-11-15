#pragma once
#include "Texture.h"

class TextureImage : public Texture {
public:
	/*
	* @param path ͼƬ·��
	*/
	TextureImage(std::string path);

	~TextureImage();

	/*
	* @brief ��ȡ������ɫ
	* @param u ��������u
	* @param v ��������v
	* @param p ����
	* @return ������ɫ
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Color* image_data;
	size_t size_x, size_y;
};

