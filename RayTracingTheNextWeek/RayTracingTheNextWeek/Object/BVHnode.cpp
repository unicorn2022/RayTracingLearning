#include "BVHnode.h"

//extern int BVH_leaf_hit;

bool BVHnode::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	// 叶节点, 则判断与当前物体是否碰撞
	if (left == nullptr && right == nullptr) {
		//BVH_leaf_hit++;
		return object->hit(r, t_min, t_max, info);
	}
	
	// 内部节点, 判断与当前包围盒是否碰撞, 进行剪枝
	if (!box.hit(r, t_min, t_max)) return false;
	

	// 递归判断左右儿子
	HitInfo left_info, right_info;
	bool left_hit = (left != nullptr) && left->hit(r, t_min, t_max, left_info);
	bool right_hit = (right != nullptr) && right->hit(r, t_min, t_max, right_info);

	if (left_hit && right_hit) {
		if (left_info.t < right_info.t) info = left_info;
		else info = right_info;
		return true;
	}
	else if (left_hit) {
		info = left_info;
		return true;
	}
	else if (right_hit) {
		info = right_info;
		return true;
	}
	else return false;
}

AABB BVHnode::GetBox() const{
	return box;
}

void BVHnode::Update() {
	if(left == nullptr && right == nullptr)
		box = object->GetBox();
	else if(left != nullptr && right == nullptr)
		box = left->GetBox();
	else if(left == nullptr && right != nullptr)
		box = right->GetBox();
	else
		box = AABB::Merge(left->GetBox(), right->GetBox());
}
