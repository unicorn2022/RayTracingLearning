#include "PDF_mixture.h"
#include "../Math/Random.h"

PDF_mixture::PDF_mixture(Ref<PDF> pdf1, Ref<PDF> pdf2, double weight1) {
	pdf[0] = pdf1;
	pdf[1] = pdf2;
	this->weight1 = weight1;
}

double PDF_mixture::value(const Vec3& direction) const {
	return weight1 * pdf[0]->value(direction) + (1 - weight1) * pdf[1]->value(direction);
}

Vec3 PDF_mixture::generate() const {
	if(Random::rand01() < weight1) return pdf[0]->generate();
	else return pdf[1]->generate();
}
