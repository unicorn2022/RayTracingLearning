#include "PDF_hit.h"

double PDF_hit::value(const Vec3& direction) const {
    return light->pdf_value(origin, direction);
}

Vec3 PDF_hit::generate() const {
    return light->random(origin);
}
