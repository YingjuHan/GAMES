// 包含必要的头文件
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <random>
#include "vec.h"

// 定义STB_IMAGE_WRITE_IMPLEMENTATION以启用图像写入功能
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

// 定义LUT纹理的分辨率
const int resolution = 128;

// 定义采样点结构体，包含方向向量和对应的概率密度函数值
typedef struct samplePoints
{
    std::vector<Vec3f> directions; // 方向向量列表
    std::vector<float> PDFs;       // 概率密度函数值列表
} samplePoints;

// 将正方形映射到余弦加权半球的采样函数
samplePoints squareToCosineHemisphere(int sample_count)
{
    samplePoints samlpeList; // 创建采样点列表
    // 计算每个维度上的采样点数
    const int sample_side = static_cast<int>(floor(sqrt(sample_count)));

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rng(0.0, 1.0);

    // 在正方形上生成采样点并映射到半球
    for (int t = 0; t < sample_side; t++)
    {
        for (int p = 0; p < sample_side; p++)
        {
            // 生成随机采样点
            double samplex = (t + rng(gen)) / sample_side;
            double sampley = (p + rng(gen)) / sample_side;

            // 将正方形采样点映射到球面坐标
            double theta = 0.5f * acos(1 - 2 * samplex); // 计算theta角度
            double phi = 2 * PI * sampley;               // 计算phi角度

            // 将球面坐标转换为笛卡尔坐标
            Vec3f wi = Vec3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

            // 计算余弦加权采样的概率密度函数值
            float pdf = wi.z / PI;

            // 将方向向量和PDF值添加到列表中
            samlpeList.directions.push_back(wi);
            samlpeList.PDFs.push_back(pdf);
        }
    }
    return samlpeList; // 返回采样点列表
}

// GGX法线分布函数(NDF)
float DistributionGGX(Vec3f N, Vec3f H, float roughness)
{
    float a = roughness * roughness;         // 计算alpha参数(roughness的平方)
    float a2 = a * a;                        // 计算alpha的平方
    float NdotH = std::max(dot(N, H), 0.0f); // 计算法线与半角向量的点积，取大于0的部分
    float NdotH2 = NdotH * NdotH;            // 点积的平方

    float nom = a2; // 分子
    // 分母计算
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom; // 完成分母计算

    // 返回NDF值，避免除零错误
    return nom / std::max(denom, 0.0001f);
}

// Schlick近似的GGX几何函数
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f; // 计算k参数

    float nom = NdotV;                    // 分子
    float denom = NdotV * (1.0f - k) + k; // 分母

    return nom / denom; // 返回几何函数值
}

// Smith几何函数
float GeometrySmith(float roughness, float NoV, float NoL)
{
    // 分别计算入射和出射方向的几何遮蔽
    float ggx2 = GeometrySchlickGGX(NoV, roughness);
    float ggx1 = GeometrySchlickGGX(NoL, roughness);

    return ggx1 * ggx2; // 返回总的几何遮蔽因子
}

// BRDF积分函数，用于计算LUT值
Vec3f IntegrateBRDF(Vec3f V, float roughness, float NdotV)
{
    float A = 0.0; // 初始化累加器
    float B = 0.0;
    float C = 0.0;
    const int sample_count = 1024;  // 采样点数量
    Vec3f N = Vec3f(0.0, 0.0, 1.0); // 法线向量(假设为z轴正方向)

    // 生成余弦加权半球采样点
    samplePoints sampleList = squareToCosineHemisphere(sample_count);

    // 对所有采样点进行积分计算
    for (int i = 0; i < sample_count; i++)
    {
        // TODO: To calculate (fr * ni) / p_o here
        Vec3f L = normalize(sampleList.directions[i]); // 归一化入射光方向
        float pdf = sampleList.PDFs[i];                // 获取PDF值
        Vec3f H = normalize(V + L);                    // 计算半角向量

        float NdotL = std::max(dot(N, L), 0.0f); // 法线与入射光方向的点积

        // 计算BRDF的各项
        float NDF = DistributionGGX(N, H, roughness);     // 法线分布函数
        float G = GeometrySmith(roughness, NdotV, NdotL); // 几何函数
        float F = 1.0f;                                   // 菲涅尔项(简化为1)

        float mu = NdotL; // 余弦项

        // 计算分子和分母
        float numerator = NDF * G * F;
        float denominator = 4.0 * NdotV * NdotL;

        // 累加积分值
        A = B = C += numerator / denominator / pdf * mu;
    }

    // 返回平均值作为最终结果
    return {A / sample_count, B / sample_count, C / sample_count};
}

// 主函数
int main()
{
    // 分配存储LUT数据的内存
    uint8_t *data = new uint8_t[resolution * resolution * 3];
    float step = 1.0 / resolution; // 计算步长

    // 遍历所有纹理像素
    for (int i = 0; i < resolution; i++)
    {
        for (int j = 0; j < resolution; j++)
        {
            // 计算粗糙度和法线与视线的夹角余弦
            float roughness = step * (static_cast<float>(i) + 0.5f);
            float NdotV = step * (static_cast<float>(j) + 0.5f);

            // 计算视线向量
            Vec3f V = Vec3f(std::sqrt(1.f - NdotV * NdotV), 0.f, NdotV);

            // 计算该像素的BRDF积分值
            Vec3f irr = IntegrateBRDF(V, roughness, NdotV);

            // 将浮点数结果转换为0-255的字节值并存储
            data[(i * resolution + j) * 3 + 0] = uint8_t(irr.x * 255.0);
            data[(i * resolution + j) * 3 + 1] = uint8_t(irr.y * 255.0);
            data[(i * resolution + j) * 3 + 2] = uint8_t(irr.z * 255.0);
        }
    }

    // 设置图像垂直翻转并写入PNG文件
    stbi_flip_vertically_on_write(true);
    stbi_write_png("GGX_E_MC_LUT.png", resolution, resolution, 3, data, resolution * 3);

    // 输出完成信息
    std::cout << "Finished precomputed!" << std::endl;
    return 0;
}