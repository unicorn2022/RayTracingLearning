#include "Perlin.h"
#include "Random.h"

double Perlin::noise(const Vec3& p) const {
	float u = p.x() - floor(p.x());
	float v = p.y() - floor(p.y());
	float w = p.z() - floor(p.z());
	int i = int(4 * p.x()) & 255;
	int j = int(4 * p.y()) & 255;
	int k = int(4 * p.z()) & 255;
	return random_value[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
}

double* Perlin::perlin_generate() {
	double* p = new double[256];
	for(int i = 0; i < 256; i++)
		p[i] = Random::rand01();
	return p;
}

int* Perlin::perlin_generate_perm() {
	int *p = new int[256];
	for(int i = 0; i < 256; i++)
		p[i] = i;
	permute(p, 256);
	return p;
}

void Perlin::permute(int p[], int n) {
	for (int i = n - 1; i > 0; i--) {
		int target = int(Random::rand01() * (i + 1));
		std::swap(p[i], p[target]);
	}
}

double* Perlin::random_value = Perlin::perlin_generate();
int* Perlin::perm_x = Perlin::perlin_generate_perm();
int* Perlin::perm_y = Perlin::perlin_generate_perm();
int* Perlin::perm_z = Perlin::perlin_generate_perm();