#pragma once

#include "Vec3.h"

class Ray {
public:
	Ray() {};
	Ray(const Point3& orig, const Vec3& dir, const double time = 0) :origin(orig), direction(dir.normalize()), time(time) {}
	Ray(const Ray& r) :origin(r.origin), direction(r.direction) {}

	Point3 Origin() const { return origin; }
	Vec3 Direction() const { return direction; }
	double Time() const { return time; }

	// P(t) = origin + t * direction
	Point3 At(double t) const {
		return origin + t * direction;
	}


private:
	Point3 origin;
	Vec3 direction;
	double time;
};