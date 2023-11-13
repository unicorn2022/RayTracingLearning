#pragma once
#include "Texture.h"

class TextureImage : public Texture {
public:
	/*
	* @param path 图片路径
	*/
	TextureImage(std::string path);

	~TextureImage();

	/*
	* @brief 获取纹理颜色
	* @param u 纹理坐标u
	* @param v 纹理坐标v
	* @param p 坐标
	* @return 纹理颜色
	*/
	virtual Color Value(double u, double v, const Point3& p) const override;

private:
	Color* image_data;
	size_t size_x, size_y;
};

