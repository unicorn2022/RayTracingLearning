#include "Isotropic.h"
#include "../Math/Random.h"

bool Isotropic::scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const {
    r_out = Ray(info.position, Random::rand_unit_sphere());
    attenuation = albedo->Value(info.u, info.v, info.position);
    return true;
}
