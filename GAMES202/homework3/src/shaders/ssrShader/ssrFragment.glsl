// 设置浮点数精度为高精度
#ifdef GL_ES
precision highp float;
#endif

// 定义uniform变量
uniform vec3 uLightDir;          // 光照方向
uniform vec3 uCameraPos;         // 相机位置
uniform vec3 uLightRadiance;     // 光照辐射度
uniform sampler2D uGDiffuse;     // 漫反射颜色G-Buffer纹理
uniform sampler2D uGDepth;       // 深度G-Buffer纹理
uniform sampler2D uGNormalWorld; // 世界空间法线G-Buffer纹理
uniform sampler2D uGShadow;      // 阴影G-Buffer纹理
uniform sampler2D uGPosWorld;    // 世界空间位置G-Buffer纹理

// 定义varying变量
varying mat4 vWorldToScreen;     // 世界到屏幕空间的变换矩阵
varying highp vec4 vPosWorld;    // 世界空间中的位置

// 定义数学常量
#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

// 一维随机数生成函数
float Rand1(inout float p) {
  p = fract(p * .1031);  // 取小数部分
  p *= p + 33.33;        // 进行变换
  p *= p + p;            // 进行变换
  return fract(p);       // 返回小数部分作为随机数
}

// 二维随机数生成函数
vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));  // 调用一维随机数生成函数生成两个随机数
}

// 初始化随机数种子
float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);  // 基于输入UV坐标生成三维向量并取小数部分
  p3 += dot(p3, p3.yzx + 33.33);           // 进行点积运算
  return fract((p3.x + p3.y) * p3.z);      // 返回最终的随机数种子
}

// 均匀采样半球函数
// 在半球面上进行均匀采样，返回一个半球面上的方向向量，同时输出该采样的概率密度函数值。
vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);              // 生成两个随机数
  float z = uv.x;                  // z坐标为第一个随机数
  float phi = uv.y * TWO_PI;       // phi角度为第二个随机数乘以2π，确定方向向量在水平面上的旋转角度
  float sinTheta = sqrt(1.0 - z*z); // 根据勾股定理，由z值计算出sinTheta，其中theta是天顶角
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z); // 球面坐标到笛卡尔坐标的转换，计算方向向量
  pdf = INV_TWO_PI;                // 概率密度函数值1/2π
  return dir;                      // 返回采样方向
}

// 余弦加权采样半球函数
vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);              // 生成两个随机数
  float z = sqrt(1.0 - uv.x);      // z坐标为sqrt(1-第一个随机数)
  float phi = uv.y * TWO_PI;       // phi角度为第二个随机数乘以2π
  float sinTheta = sqrt(uv.x);     // 计算sin(theta)
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z); // 计算方向向量
  pdf = z * INV_PI;                // 计算概率密度函数值
  return dir;                      // 返回采样方向
}

// 构建局部坐标系函数
void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);         // 获取法线z分量的符号
  if (n.z == 0.0) {
    sign_ = 1.0;                   // 如果z分量为0，则符号设为1
  }
  float a = -1.0 / (sign_ + n.z);  // 计算辅助变量a
  float b = n.x * n.y * a;         // 计算辅助变量b
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x); // 计算第一个基向量
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y); // 计算第二个基向量
}

// 投影函数
vec4 Project(vec4 a) {
  return a / a.w;  // 进行透视除法
}

// 获取世界空间点的深度
float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w; // 变换到屏幕空间并获取深度
  return depth;
}

/*
 * 将世界空间中的点变换到屏幕空间([0, 1] x [0, 1])
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5; // 投影并变换到[0,1]范围
  return uv;
}

// 获取G-Buffer中的深度值
float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;  // 从深度纹理中采样
  if (depth < 1e-2) {
    depth = 1000.0;  // 如果深度值过小，则设为一个大值
  }
  return depth;
}

// 获取G-Buffer中的世界空间法线
vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;  // 从法线纹理中采样
  return normal;
}

// 获取G-Buffer中的世界空间位置
vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;  // 从位置纹理中采样
  return posWorld;
}

// 获取G-Buffer中的阴影值
float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;  // 从阴影纹理中采样
  return visibility;
}

// 获取G-Buffer中的漫反射颜色
vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;  // 从漫反射纹理中采样
  diffuse = pow(diffuse, vec3(2.2));            // 进行伽马校正
  return diffuse;
}

/*
 * 计算漫反射BSDF值
 *
 * wi, wo 都在世界空间中
 * uv 在屏幕空间中，范围为[0, 1] x [0, 1]
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 L = vec3(0.0);  // 初始化光照值为0
  vec3 albedo = GetGBufferDiffuse(uv);
  vec3 normal = GetGBufferNormalWorld(uv);
  float cosTheta = dot(wi, normal);
  if (cosTheta > 0.0) {
    L = albedo * INV_PI * cosTheta;
  }
  return L;
}

/*
 * 计算带阴影的方向光
 * uv 在屏幕空间中，范围为[0, 1] x [0, 1]
 */
vec3 EvalDirectionalLight(vec2 uv) {
  vec3 Le = vec3(0.0);  // 初始化光照值为0
  Le = GetGBufferuShadow(uv) * uLightRadiance;
  return Le;
}

// 光线步进函数
bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  float step = 0.05;
  const int totalStepTimes = 150;

  vec3 oneStep = normalize(dir) * step;
  vec3 curPos = ori;
  for (int curStepTimes = 0; curStepTimes < totalStepTimes; curStepTimes++) {
    vec2 screenUV = GetScreenCoordinate(curPos);
    float rayDepth = GetDepth(curPos);
    float gBufferDepth = GetGBufferDepth(screenUV);
    if (rayDepth - gBufferDepth > 0.0001) {
      hitPos = curPos;
      return true;
    }
    curPos += oneStep;
  }
  return false;
}

// SSR
vec3 EvalSSReflect(vec3 wi, vec3 wo, vec2 uv) {
  vec3 normal = GetGBufferNormalWorld(uv);
  vec3 reflectDir = normalize(reflect(-wo, normal));
  vec3 hitPos;
  if (RayMarch(vPosWorld.xyz, reflectDir, hitPos)) {
    vec2 screenUV = GetScreenCoordinate(hitPos);
    return GetGBufferDiffuse(screenUV);
  }
  return vec3(0.0);
}

// 定义采样数量
#define SAMPLE_NUM 1

/**
 * 主函数 EvalDiffuse * EvalDirectionalLight
 */
 /*
void main() {
  float s = InitRand(gl_FragCoord.xy);  // 初始化随机数种子
 
  vec3 L = vec3(0.0);  // 初始化光照值为0

  // 获取当前片元的世界空间位置对应的屏幕坐标
  vec3 worldPos = vPosWorld.xyz;
  vec2 screenUV = GetScreenCoordinate(worldPos);

  // 入射光方向
  vec3 wi = normalize(uLightDir);

  // 出射光方向
  vec3 wo = normalize(uCameraPos - worldPos);

  // 直接光照
  L = EvalDiffuse(wi, wo, screenUV) * EvalDirectionalLight(screenUV);
  
  // 对光照值进行伽马校正
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  
  // 设置片元颜色
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
}
*/

/**
 *  主函数 EvalSSReflect 测试
 */
 /*
 void main() {
  float s = InitRand(gl_FragCoord.xy);  // 初始化随机数种子
  vec3 L = vec3(0.0);  // 初始化光照值为0
  vec3 worldPos = vPosWorld.xyz;
  vec2 screenUV = GetScreenCoordinate(worldPos);
  vec3 wi = normalize(uLightDir);
  vec3 wo = normalize(uCameraPos - worldPos);

  // SSR
  L = 0.5 * (GetGBufferDiffuse(screenUV) + EvalSSReflect(wi, wo, screenUV));

  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
 }
*/

/**
 * ssr
 */

 void main() {
  float s = InitRand(gl_FragCoord.xy);  // 初始化随机数种子
 
  vec3 L = vec3(0.0);  // 初始化光照值为0

  // 获取当前片元的世界空间位置对应的屏幕坐标
  vec3 worldPos = vPosWorld.xyz;
  vec2 screenUV = GetScreenCoordinate(worldPos);

  // 入射光方向
  vec3 wi = normalize(uLightDir);

  // 出射光方向
  vec3 wo = normalize(uCameraPos - worldPos);

  // 直接光照
  L = EvalDiffuse(wi, wo, screenUV) * EvalDirectionalLight(screenUV);

  vec3 L_indirect = vec3(0.0);
  for (int i = 0; i < SAMPLE_NUM; i++) {
    float pdf;
    vec3 localDir = SampleHemisphereCos(s,pdf);
    vec3 normal = GetGBufferNormalWorld(screenUV);
    vec3 b1, b2;
    LocalBasis(normal, b1, b2);
    vec3 dir = normalize(mat3(b1, b2, normal) * localDir);

    vec3 position_1;
    if (RayMarch(worldPos, dir, position_1)) {
      vec2 hitScreenUV = GetScreenCoordinate(position_1);
      L_indirect += EvalDiffuse(dir, wo, screenUV) / pdf * EvalDiffuse(wi, dir, hitScreenUV) * EvalDirectionalLight(hitScreenUV);
    }
  }

  L_indirect /= float(SAMPLE_NUM);
  L = L + L_indirect;
  
  // 对光照值进行伽马校正
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  
  // 设置片元颜色
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
}