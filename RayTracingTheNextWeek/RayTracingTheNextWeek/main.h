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