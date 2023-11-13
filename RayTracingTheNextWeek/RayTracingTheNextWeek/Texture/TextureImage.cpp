#include "TextureImage.h"
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#include "../3rdpatry/stb_image.h"

TextureImage::TextureImage(std::string path) {
	int nx, ny, nn;
	unsigned char* tex_data = stbi_load(path.c_str(), &nx, &ny, &nn, 0);
	if(tex_data == nullptr) {
		std::cerr << "ERROR: Í¼Æ¬¼ÓÔØÊ§°Ü, Â·¾¶Îª: " << path << ".\n";
		exit(-1);
	}

	size_x = nx;
	size_y = ny;
	image_data = new Color[size_x * size_y];
	for(int i = 0; i < size_x; i++)
		for (int j = 0; j < size_y; j++) {
			double r = tex_data[nn * (i + j * size_x)] / 255.0;
			double g = tex_data[nn * (i + j * size_x) + 1] / 255.0;
			double b = tex_data[nn * (i + j * size_x) + 2] / 255.0;
			image_data[i + j * size_x] = Color(r, g, b);
		}
}

TextureImage::~TextureImage() {
	delete[] image_data;
}

Color TextureImage::Value(double u, double v, const Point3& p) const {
	int i = std::clamp<int>(u * size_x, 0, size_x - 1);
	int j = std::clamp<int>(v * size_y, 0, size_y - 1);
	return image_data[i + j * size_x];
}
