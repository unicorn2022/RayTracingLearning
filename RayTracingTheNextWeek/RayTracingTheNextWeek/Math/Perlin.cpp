#include "Perlin.h"
#include "Random.h"

double Perlin::noise(const Point3& p) const {
	int i = floor(p.x());
	int j = floor(p.y());
	int k = floor(p.z());

	double u = p.x() - i;
	double v = p.y() - j;
	double w = p.z() - k;

	Point3 list[2][2][2];
	for(int a = 0; a < 2; a++)
		for(int b = 0; b < 2; b++)
			for(int c = 0; c < 2; c++)
				list[a][b][c] = random_value[
					perm_x[(i + a) & 255] ^
					perm_y[(j + b) & 255] ^
					perm_z[(k + c) & 255]
				];
	return trilinear_interp(list, u, v, w);
}

double Perlin::turb(const Point3& p, int depth) const {
	double accumulate = 0;
	Point3 t = p;
	double weight = 1.0;
	for (int i = 0; i < depth; i++) {
		accumulate += weight * noise(t);
		weight *= 0.5;
		t *= 2;
	}
	return abs(accumulate);
}

double Perlin::trilinear_interp(Point3 list[2][2][2], double u, double v, double w) const {
	double uu = u * u * (3 - 2 * u);
	double vv = v * v * (3 - 2 * v);
	double ww = w * w * (3 - 2 * w);
	
	double accumulate = 0;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				Point3 weight(u - i, v - j, w - k);
				accumulate +=
				(i * uu + (1 - i) * (1 - uu)) *
				(j * vv + (1 - j) * (1 - vv)) *
				(k * ww + (1 - k) * (1 - ww)) * list[i][j][k].dot(weight);
			}
	return accumulate;
}

Point3* Perlin::perlin_generate() {
	Point3* p = new Point3[256];
	for (int i = 0; i < 256; i++)
		p[i] = Point3(
			-1 + 2 * Random::rand01(),
			-1 + 2 * Random::rand01(),
			-1 + 2 * Random::rand01()
		).normalize();
	return p;
}

int* Perlin::perlin_generate_perm() {
	int *p = new int[256];
	for(int i = 0; i < 256; i++)
		p[i] = i;
	for (int i = 255; i > 0; i--) {
		int target = int(Random::rand01() * (i + 1));
		std::swap(p[i], p[target]);
	}
	return p;
}

Point3* Perlin::random_value = Perlin::perlin_generate();
int* Perlin::perm_x = Perlin::perlin_generate_perm();
int* Perlin::perm_y = Perlin::perlin_generate_perm();
int* Perlin::perm_z = Perlin::perlin_generate_perm();