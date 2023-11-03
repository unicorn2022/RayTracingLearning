#pragma once

#include "vec3.h"

class ray {
public:
	ray() {};
	ray(const point3& orig, const vec3& dir) :origin(orig), direction(dir) {}

	point3 getOrigin() const { return origin; }
	vec3 getDirection() const { return direction; }

	// P(t) = origin + t * direction
	point3 at(double t) const {
		return origin + t * direction;
	}


private:
	point3 origin;
	vec3 direction;
};