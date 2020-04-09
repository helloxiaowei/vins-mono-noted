#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
* @class 1. FeaturePerFrame
* @brief _point 每帧的特征点[x,y,z,u,v,vx,vy]
* @brief td IMU和cam同步时间差
* detailed 
*/
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;
    Vector2d uv;
    Vector2d velocity;

    double z; // 特征点的深度
    bool is_used;// 是否被用了
    double parallax;// 视差
    MatrixXd A; //变换矩阵
    VectorXd b;
    double dep_gradient; // ？？？
};

/**
* @class 2. FeaturePerId，某feature_id下的所有FeaturePerFrame
* @brief feature_id 特征点ID
* @brief start_frame 出现该角点的第一帧的id--start_frame
* detailed 
*/
class FeaturePerId
{
  public:
    const int feature_id;// 特征点ID索引
    int start_frame;// 首次被观测到时，该帧的索引
    vector<FeaturePerFrame> feature_per_frame; // 能够观测到某个特征点的所有相关帧

    int used_num;// 该特征被观测到的次数
    bool is_outlier;// 是否外点
    bool is_margin;// 是否Marg边缘化
    double estimated_depth; // 估计的逆深度
    int solve_flag; // 求解器 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p; // ？？？

    FeaturePerId(int _feature_id, int _start_frame) 
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();// 返回最后一次观测到这个特征点的帧数ID
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);// 设置旋转矩阵
    void clearState();// 清理特征管理器中的特征feature,list<FeaturePerId> feature
    int getFeatureCount(); //窗口中被跟踪的特征数量
    // 特征点进入时检查视差
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);// 两帧间对应的特征点

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);//设置特征点的逆深度估计值
    void removeFailures(); // 剔除feature中估计失败的点（solve_flag == 2）
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]); // 对特征点进行三角化求深度（SVD分解）
    //边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();//边缘化最老帧时，直接将特征点所保存的帧号向前滑动
    void removeFront(int frame_count);//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
    void removeOutlier();
    list<FeaturePerId> feature;// 重要！！！ 通过FeatureManager可以得到滑动窗口内所有的角点信息
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;// 旋转矩阵
    Matrix3d ric[NUM_OF_CAM]; // 所有的旋转矩阵
};

#endif