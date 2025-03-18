#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

#include "rrf.h"

void draw_histogram(const std::vector<int> &counts, double mu, double kappa, int num_bins)
{
    int max_count = *max_element(counts.begin(), counts.end());
    double scale = 50.0 / (max_count + 1); // 柱状图最大长度50个星号

    std::cout << "μ = " << mu << " | κ = " << kappa << std::endl;
    for (int i = 0; i < num_bins; ++i)
    {
        double start = 2 * M_PI * i / num_bins;
        double end = 2 * M_PI * (i + 1) / num_bins;
        std::cout << std::fixed << std::setprecision(2)
                  << "[" << start << ", " << end << "): ";

        // 绘制星号
        for (int j = 0; j < counts[i] * scale; ++j)
        {
            std::cout << '*';
        }
        std::cout << " (" << counts[i] << ")" << std::endl;
    }
    std::cout << std::endl;
}

// 生成 n 维模长为 1 的随机向量
void generateRandomUnitVector(int n, std::vector<double> &random_point, std::mt19937 &gen)
{
    // 随机数生成器
    // static std::mt19937 gen(rd);               // Mersenne Twister 引擎
    std::normal_distribution<> dist(0.0, 1.0); // 标准正态分布 (均值 0，标准差 1)

    // 生成 n 维随机点
    // std::vector<double> random_point;
    random_point.resize(n);
    for (int i = 0; i < n; ++i)
        random_point[i] = dist(gen);

    // 计算模长
    double norm = 0.0;
    for (double x : random_point)
        norm += x * x;
    norm = std::sqrt(norm);

    // 归一化
    for (double &x : random_point)
        x /= norm;
}

// 计算两个向量的点积
double dotProduct(const std::vector<double> &a, const std::vector<double> &b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

// 归一化向量
void normalizeVector(std::vector<double> &v)
{
    double norm = 0.0;
    for (double x : v)
        norm += x * x;
    norm = std::sqrt(norm);
    // 防止除以零
    if (norm == 0.0)
        throw std::runtime_error("Cannot normalize a zero vector.");

    for (size_t i = 0; i < v.size(); ++i)
        v[i] = v[i] / norm;
}

double fast_vonmises(double mu, double kappa, std::mt19937 &gen)
{
    // static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dis(0.0, 1.0);
    static std::normal_distribution<double> norm(0.0, 1.0);

    if (kappa == 0.0)
    {
        return 2.0 * M_PI * dis(gen); // 纯均匀分布
    }

    // 快速近似处理
    if (kappa < 1.0)
    { // 低集中度近似
        double theta = mu + (dis(gen) - 0.5) * 2.0 * M_PI;
        return std::fmod(theta + 2.0 * M_PI, 2.0 * M_PI); // 保证[0, 2π)
    }

    // 高集中度正态分布近似
    double sigma = 1.0 / std::sqrt(kappa); // 标准差公式
    double theta = mu + norm(gen) * sigma;
    theta = std::fmod(theta, 2.0 * M_PI);
    return theta < 0 ? theta + 2.0 * M_PI : theta;
}

// 计算向量 b 在向量 a 的垂直分量，并归一化
std::vector<double> perpendicularComponentNormalized(const std::vector<double> &a, const std::vector<double> &b)
{
    // 点积和模长平方
    double dot_ab = dotProduct(a, b);
    double dot_aa = dotProduct(a, a);

    // 投影分量
    std::vector<double> proj_b_on_a(a.size());
    for (size_t i = 0; i < a.size(); ++i)
    {
        proj_b_on_a[i] = (dot_ab / dot_aa) * a[i];
    }

    // 垂直分量
    std::vector<double> b_perpendicular(b.size());
    for (size_t i = 0; i < b.size(); ++i)
    {
        b_perpendicular[i] = b[i] - proj_b_on_a[i];
    }

    // 归一化垂直分量
    normalizeVector(b_perpendicular);
    return b_perpendicular;
}

// Von Mises-Fisher 分布随机采样函数
void vonMisesFisher(int n, const std::vector<double> &vec_O, double kappa, std::vector<double> &vec_new, std::mt19937 &gen)
{
    // vec_new.resize(2);
    double th = fast_vonmises(0, 1, gen);
    std::vector<double> random_point;
    generateRandomUnitVector(n, random_point, gen);
    std::vector<double> va = vec_O;
    normalizeVector(va);
    std::vector<double> vb = perpendicularComponentNormalized(va, random_point);

    double sin_th = std::sin(th);
    double cos_th = std::cos(th);
    for (int32_t i = 0; i < n; i++)
    {
        vec_new[i] = sin_th * vb[i] + cos_th * va[i];
    }
}