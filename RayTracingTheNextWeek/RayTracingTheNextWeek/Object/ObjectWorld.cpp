#include "ObjectWorld.h"
#include "../config.h"

#include <ctime>

//extern int AABB_hit;
//extern int BVH_leaf_hit;
//extern int BVH_node_cnt;

bool ObjectWorld::hit(const Ray& r, double t_min, double t_max, HitInfo& info) const {
	//auto t = clock();
	//return root->hit(r, t_min, t_max, info);
	
	//std::cerr << "AABB_hit = " << AABB_hit << std::endl;
	//std::cerr << "BVH_leaf_hit = " << BVH_leaf_hit << std::endl;
	//std::cerr << "total hit = " << AABB_hit + BVH_leaf_hit << std::endl;
	//std::cerr << "BVH_node_cnt = " << BVH_node_cnt << std::endl;
	//std::cerr << "BVH info.t = " << info.t << " ";
	//std::cerr << "BVH hit time: " << clock() - t << "\n\n";
	//t = clock();

	bool hit_anything = false;
	HitInfo temp_info;
	double closest_so_far = t_max; // 获取 ray 相交的最小的 t

	for (const auto& object : objects) {
		if (object->hit(r, t_min, closest_so_far, temp_info)) {
			hit_anything = true;
			closest_so_far = temp_info.t;
			info = temp_info;
		}
	}
	
	
	//std::cerr << "object cnt = " << objects.size() << std::endl;
	//std::cerr << "info.t = " << info.t << " ";
	//std::cerr << "hit time: " << clock() - t << std::endl;
	//exit(0);

	return hit_anything;
}

Color ObjectWorld::GetColor(const Ray& r, int depth) {
	HitInfo record;

	// 如果碰撞到了, 则根据材质计算反射光线
	if (this->hit(r, 0, INFINITY, record)) {
		Ray scattered;
		Color attenuation;
		Color emit = record.material->emitted(record.u, record.v, record.position);
		if (depth < max_depth && record.material->scatter(r, record, attenuation, scattered))
			return emit + attenuation * GetColor(scattered, depth + 1);
		else
			return emit;
	}
	// 如果不相交, 则返回黑色
	else {
		return Color(1.0f);
	}
	/*
	// 如果不相交, 则根据方向插值背景颜色
	else {
		Vec3 direction_unit = r.Direction().normalize();
		double t = 0.5 * (direction_unit.y() + 1);
		return (1 - t) * Color(1.0f) + t * background;
	}*/
}

void ObjectWorld::Build() {
	int size = objects.size();

	// 递归构建 BVH
	root = New<BVHnode>();
	build(root, 0, size - 1);
}

void ObjectWorld::build(Ref<BVHnode> u, int L, int R, int deep) {
	/*std::sort(objects.begin() + L, objects.begin() + R + 1, [&](const Ref<ObjectBase>& a, const Ref<ObjectBase>& b) {
		return a->GetBox().Min()[deep % 3] < b->GetBox().Min()[deep % 3];
	});*/
	//BVH_node_cnt++;

	u->L = L; u->R = R;
	int size = R - L + 1;
	if (size == 1) {
		u->left = u->right = nullptr;
		u->object = objects[L];
	}
	else {
		int mid = (L + R) >> 1;
		u->left = New<BVHnode>();
		u->right = New<BVHnode>();
		u->object = nullptr;
		build(u->left, L, mid);
		build(u->right, mid + 1, R);
	}
	u->Update();

	//std::cerr << "L = " << L << ",\tR = " << R << std::endl;
	//std::cerr << "u->box = (" << u->GetBox().Min() << "),\t(" << u->GetBox().Max() <<")\n";
	
}
