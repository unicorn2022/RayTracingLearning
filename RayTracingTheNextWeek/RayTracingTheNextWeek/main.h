#pragma once
#include <iostream>
#include <iomanip>
#include <ctime>
#include <thread>
#include <algorithm>

#include "Camera/Camera.h"
#include "Math/Random.h"

#include "Object/Box.h"
#include "Object/ObjectWorld.h"
#include "Object/RectXY.h"
#include "Object/RectXZ.h"
#include "Object/RectYZ.h"
#include "Object/Sphere.h"
#include "Object/SphereMoving.h"
#include "Object/Transform/FlipNormal.h"
#include "Object/Transform/Translate.h"
#include "Object/Transform/RotateY.h"
#include "Object/Transform/ConstantMedium.h"

#include "Material/Lambertian.h"
#include "Material/Dielectric.h"
#include "Material/Metal.h"
#include "Material/Emit.h"

#include "Texture/TextureConstant.h"
#include "Texture/TextureChecker.h"
#include "Texture/TextureNoise.h"
#include "Texture/TextureImage.h"

#include "config.h"
#include "Utils/Utils.h"

#include <string>
#include <Windows.h>
#include <fstream>

enum class ConsoleColor {
	Clear,	// 原色
	White,	// 白色
	Red,	// 红色
	Green,	// 绿色
	Blue,	// 蓝色
	Yellow,	// 黄色
	Pink,	// 粉色
	Cyan	// 青色
};

static char SetConsoleColor(ConsoleColor color) {
	HANDLE hdl = GetStdHandle(STD_OUTPUT_HANDLE);
	switch (color) {
	case ConsoleColor::Clear:
		SetConsoleTextAttribute(hdl, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED);
		break;
	case ConsoleColor::White:
		SetConsoleTextAttribute(hdl, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Red:
		SetConsoleTextAttribute(hdl, FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Green:
		SetConsoleTextAttribute(hdl, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Blue:
		SetConsoleTextAttribute(hdl, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Yellow:
		SetConsoleTextAttribute(hdl, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Pink:
		SetConsoleTextAttribute(hdl, FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
		break;
	case ConsoleColor::Cyan:
		SetConsoleTextAttribute(hdl, FOREGROUND_BLUE | FOREGROUND_GREEN | FOREGROUND_INTENSITY);
		break;
	default:
		break;
	}
	return '\0';
}

static void PrintPercent(int now, int total) {
	std::cerr << "(";
	if (now == total) {
		SetConsoleColor(ConsoleColor::Green);
		std::cerr << "success";
		SetConsoleColor(ConsoleColor::Clear);
	}
	else {
		SetConsoleColor(ConsoleColor::Red);
		std::cerr << now * 100 / total << "%";
		SetConsoleColor(ConsoleColor::Clear);
	}
	std::cerr << ") ";
}