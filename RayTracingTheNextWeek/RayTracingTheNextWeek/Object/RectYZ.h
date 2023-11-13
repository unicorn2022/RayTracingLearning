#pragma once
#include "Object.h"
class RectYZ : public Object {
public:
	/*
	* @param y1 y ����Сֵ
	* @param y2 y �����ֵ
	* @param z1 z ����Сֵ
	* @param z2 z �����ֵ
	* @param k	x ��ֵ
	* @param material ����
	*/
	RectYZ(double y1, double y2, double z1, double z2, double k, Ref<Material> material)
		: z1(z1), z2(z2), y1(y1), y2(y2), k(k), material(material) {}

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
	double y1, y2, z1, z2, k;
};

