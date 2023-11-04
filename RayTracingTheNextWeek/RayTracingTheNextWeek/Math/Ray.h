#pragma once

#include "Vec3.h"

class Ray {
public:
	Ray() {};
	Ray(const Point3& orig, const Vec3& dir) :origin(orig), direction(dir) {}

	Point3 Origin() const { return origin; }
	Vec3 Direction() const { return direction; }

	// P(t) = origin + t * direction
	Point3 At(double t) const {
		return origin + t * direction;
	}


private:
	Point3 origin;
	Vec3 direction;
};