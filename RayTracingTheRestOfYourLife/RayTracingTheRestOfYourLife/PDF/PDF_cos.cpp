#include "PDF_cos.h"
#include "../Math/Random.h"

double PDF_cos::value(const Vec3& direction) const {
	double cosine = direction.normalize().dot(uvw.w());
	if(cosine > 0) return cosine / PI;
	else return 0;
}

Vec3 PDF_cos::generate() const {
	return uvw.local(Random::rand_cosine_direction());
}
