#pragma once
#include "Object.h"
class RectXZ : public Object {
public:
	/*
	* @param x1 x ����Сֵ
	* @param x2 x �����ֵ
	* @param z1 z ����Сֵ
	* @param z2 z �����ֵ
	* @param k	y ��ֵ
	* @param material ����
	*/
	RectXZ(double x1, double x2, double z1, double z2, double k, Ref<Material> material)
		: x1(x1), x2(x2), z1(z1), z2(z2), k(k), material(material) {}

	/*
	* @brief �жϹ����Ƿ��뵱ǰ������ײ
	* @param r ����
	* @param t_min ���ߵ���С t ֵ
	* @param t_max ���ߵ���� t ֵ
	* @param info ��ײ����Ϣ
	* @return �Ƿ���ײ
	*/
	virtual bool hit(const Ray& r, double t_min, double t_max, HitInfo& info) const override;

	/*
	* @brief ��ȡ��ǰ����İ�Χ��
	*/
	virtual AABB GetBox() const override;

private:
	Ref<Material> material;
	double x1, x2, z1, z2, k;
};

