#pragma once
#include "Material.h"
#include "../config.h"
#include "../Texture/Texture.h"

class Emit : public Material {
public:
	/*
	* @param emit �Է�������
	*/
	Emit(Ref<Texture> emit) : emit(emit) {}

	/*
	* @brief ����ɢ�����
	* @param r_in �������
	* @param info ��ײ��Ϣ
	* @param attenuation ������ɢ��ʱ, ��ǿ���˥��, ��Ϊrgb��������
	* @param r_out ɢ�����
	* @return �Ƿ�õ���ɢ�����
	*/
	virtual bool scatter(const Ray& r_in, const HitInfo& info, Color& attenuation, Ray& r_out) const override;

	/*
	* @brief �Է���
	* @param u uv����
	* @param v uv����
	* @param p ��ײ��
	*/
	virtual Color emitted(double u, double v, const Point3& p) const override;

private:
	Ref<Texture> emit;
};

