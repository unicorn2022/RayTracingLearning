#pragma once
#include "Vec3.h"

class ONB {
public:
	/*
	* @brief 根据法线构建局部坐标系
	* @param normal 法线
	*/
	ONB(Vec3 normal);

	/*
	* @brief 获取局部坐标对应的世界坐标
	*/
	Vec3 local(double x, double y, double z) const;

	/*
	* @brief 获取局部坐标对应的世界坐标
	*/
	Vec3 local(const Vec3& v) const;

private:
	/*
	* @brief 根据法线构建局部坐标系
	* @param normal 法线
	*/
	void build_from_w(const Vec3& normal);


public:
	const Vec3& operator[](int index) const { return axis[index]; }
	const Vec3& u() const { return axis[0]; }
	const Vec3& v() const { return axis[1]; }
	const Vec3& w() const { return axis[2]; }

private:
	Vec3 axis[3];
};

