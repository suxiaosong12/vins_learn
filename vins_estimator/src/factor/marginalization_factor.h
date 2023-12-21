#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    // 构造函数需要，cost function(约束)，loss function：残差的计算方式，相关联的参数块，待边缘化的参数块的索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);  // 添加残差块相关信息（优化变量，待marg的变量）
    void preMarginalize();  // 计算每个残差对应的雅可比，并更新parameter_block_data
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);  // 获取参数块指针

    std::vector<ResidualBlockInfo *> factors;  // 所有残差块信息
    int m, n;  // m为要边缘化的变量个数，n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size; // 每个参数块的大小
    int sum_block_size;  // 所有参数块大小之和
    std::unordered_map<long, int> parameter_block_idx; // 被边缘化的参数块对应的id <优化变量内存地址,在矩阵中的id>
    std::unordered_map<long, double *> parameter_block_data;  // 每个参数块的数据 <优化变量内存地址,数据>
    // 他们的key都同一是long类型的内存地址，而value分别是，各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据

    std::vector<int> keep_block_size; // 保留的参数块大小
    std::vector<int> keep_block_idx;  // 保留的参数块对应的id
    std::vector<double *> keep_block_data;  // 保留的参数块数据
    // 他们是进行边缘化之后保留下来的各个优化变量的长度，各个优化变量在id以各个优化变量对应的double指针类型的数据

    Eigen::MatrixXd linearized_jacobians;  //边缘化之后从信息矩阵恢复出来的雅克比矩阵
    Eigen::VectorXd linearized_residuals;  //边缘化之后从信息矩阵恢复出来的残差向量
    const double eps = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;  // 计算当前因子的残差以及对应的雅可比矩阵

    MarginalizationInfo* marginalization_info;  // 该指针指向包含当前要被边缘化的变量和对应因子的信息的 MarginalizationInfo 对象
};
