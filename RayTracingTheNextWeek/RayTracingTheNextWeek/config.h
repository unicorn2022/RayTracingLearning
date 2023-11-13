﻿#pragma once

/* 画质设置 */
static const int samples_per_pixel = 100;	// 每个像素的采样次数
static const int max_depth = 10;			// 最大递归深度

/* 线程设置 */
static const int thread_cnt = 7;			// 线程数

#define Ref std::shared_ptr
#define New std::make_shared