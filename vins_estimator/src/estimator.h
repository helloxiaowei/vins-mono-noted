#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

/**
* @class Estimator 状态估计器
* @Description IMU预积分，图像IMU融合的初始化和状态估计，重定位
* detailed 
*/
class Estimator
{
  public:
    Estimator();

    void setParameter();// 设置参数

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);// 处理IMU
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);// 处理Image
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);// 重定位

    // internal
    void clearState(); // 清除状态
    bool initialStructure();// 初始化结构
    bool visualInitialAlign(); // VIO几何
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l); // 相对位姿
    
    void slideWindow();// 滑动窗
    void solveOdometry();// 里程计
    void slideWindowNew();// 新旧滑动窗
    void slideWindowOld();
    void optimization(); // 重要！！！优化
    void vector2double();
    void double2vector();
    bool failureDetection(); // 检测失败

    //Solver标志：初始化、非线性
    enum SolverFlag 
    {
        INITIAL,// 
        NON_LINEAR
    };
    //边缘化Marg标志：边缘旧的、边缘化第二个新帧
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;

    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    //窗口中的[P,V,R,Ba,Bg]
    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;// Cam和IMU的时间差
    // back_R0、back_P0为窗口中最老帧的位姿
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];// ROS的消息类型

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];// 预积分类
    Vector3d acc_0, gyr_0;

    //窗口中的dt,a,v
    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;// 帧数量
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;//特征管理器类
    MotionEstimator m_estimator; // 姿态估计类
    InitialEXRotation initial_ex_rotation;// 初始化Cam到IMU的外参数

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    // Optimization函数的模块
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;// 边缘化先验值
    vector<double *> last_marginalization_parameter_blocks;

    //kay是时间戳，val是图像帧
    //图像帧中保存了图像帧的特征点、时间戳、位姿Rt，预积分对象pre_integration，是否是关键帧。
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration; 

    //重定位所需的变量
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
