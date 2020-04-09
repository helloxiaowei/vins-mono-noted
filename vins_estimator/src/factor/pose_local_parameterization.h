#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;// 计算雅克比
    virtual int GlobalSize() const { return 7; }; // 全局大小
    virtual int LocalSize() const { return 6; }; // 局部大小
};
