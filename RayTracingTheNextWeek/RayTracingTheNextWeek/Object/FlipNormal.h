#pragma once
#include "Object.h"
class FlipNormal : public Object {
public:
	FlipNormal(Ref<Object> object) : object(object) {}

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
	Ref<Object> object;
};

