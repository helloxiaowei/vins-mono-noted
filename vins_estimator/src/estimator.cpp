#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

//视觉测量残差的协方差矩阵
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

//清空或初始化滑动窗口中所有的状态量               
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   处理IMU数据
 * @Description IMU预积分，中值积分得到当前PQV作为优化初值
 * @param[in]   dt 时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 1.imu未进来数据
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    // 2.IMU 预积分类对象还没出现，创建一个
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    
    if (frame_count != 0)
    {   // 3.预积分操作
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        // 4.dt、加速度、角速度加到buf中
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count; 
        // 5.采用的是中值积分的传播方式        
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;// a0=Q(a^-ba)-g 已知上一帧imu速度
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];// w=0.5(w0+w1)-bg
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;// a1 当前帧imu速度
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);// 中值积分下的加速度a=1/2(a0+a1)
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;// P=P+v*t+1/2*a*t^2
        Vs[j] += dt * un_acc;// V=V+a*t
    }
    // 6.更新上一帧的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧               
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 1. 通过检测两帧之间的视差决定次新帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))//添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
        marginalization_flag = MARGIN_OLD;//=0
    else
        marginalization_flag = MARGIN_SECOND_NEW;//=1

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // 2. 填充imageframe的容器以及更新临时预积分初始值
    ImageFrame imageframe(image, header.stamp.toSec());//ImageFrame类包括特征点、时间、位姿Rt、预积分对象pre_integration、是否关键帧
    imageframe.pre_integration = tmp_pre_integration;// tmp_pre_integration是之前IMU 预积分计算的
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));// map<double, ImageFrame> all_image_frame;
    
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};//更新临时预积分初始值

    // 3. 如果没有外参则标定IMU到相机的外参
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            //得到两帧之间归一化特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //标定从camera到IMU之间的外参数
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;// 完成外参标定
            }
        }
    }
    // 4. 初始化
    if (solver_flag == INITIAL)//初始化
    {
        //frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE=10
        //确保有足够的frame参与初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            //有外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               // 4.1 重要！！！ 视觉SFM初始化
               result = initialStructure();
               // 4.2 更新初始化时间戳
               initial_timestamp = header.stamp.toSec();
            }
            if(result)//初始化成功
            {
                // 先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                solver_flag = NON_LINEAR;// 初始化更改为非线性
                solveOdometry(); // 4.3 非线性化求解里程计
                slideWindow();// 4.4 滑动窗
                f_manager.removeFailures();// 4.5 剔除feature中估计失败的点（solve_flag == 2）0 haven't solve yet; 1 solve succ; 2 solve fail;
                ROS_INFO("Initialization finish!");
                // 4.6 初始化窗口中PVQ
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();// 初始化失败则直接滑动窗口
        }
        else
            frame_count++;// 图像帧数量+1
    }
    else// 5. 紧耦合的非线性优化
    {
        TicToc t_solve;
        solveOdometry();// 5.1 非线性化求解里程计
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        //5.2 故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();// 清空状态
            setParameter();// 重设参数
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow(); // 5.3 滑动窗口
        f_manager.removeFailures();// 5.4 移除失败的
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);// 关键位姿的位置 Ps

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE]; 
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief   视觉的结构初始化，首先初始化Vision-only SFM，然后初始化Visual-Inertial Alignment
 * @Description 确保IMU有充分运动激励
 *              relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *              sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *              visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
*/
bool Estimator::initialStructure()
{
    TicToc t_sfm;

    // 1. 通过加速度标准差判断IMU是否有充分运动以初始化。
    {
        map<double, ImageFrame>::iterator frame_it;// 迭代器
        // 1.1 先求均值 aver_g
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;// a=v/t
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);// 总帧数减1，因为all_image_frame第一帧不算
        // 1.2 再求标准差var,表示线加速度的标准差
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 判断，加速度标准差大于0.25则代表imu充分激励，足够初始化
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }

    // 2. 将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    Quaterniond Q[frame_count + 1];//旋转四元数Q
    Vector3d T[frame_count + 1]; // 平移矩阵T
    map<int, Vector3d> sfm_tracked_points; //存储SFM重建出特征点的坐标
    vector<SFMFeature> sfm_f;// SFMFeature三角化状态、特征点索引、平面观测、位置坐标、深度

    for (auto &it_per_id : f_manager.feature)// list<FeaturePerId> feature滑动窗口内所有角点
    {
        int imu_j = it_per_id.start_frame - 1;// 第一次观测到特征点的帧数-1
        SFMFeature tmp_feature;
        tmp_feature.state = false;// 状态state
        tmp_feature.id = it_per_id.feature_id;// 特征点ID
        // observation
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;// 3D特征点坐标
            // 所有观测到该特征点的图像帧ID和图像坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);  
    } 

    Matrix3d relative_R;
    Vector3d relative_T;
    int l;

    // 3. 返回滑动窗口中第一个满足视差的帧，为l帧，以及RT,可以三角化。
    // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    //此处的relative_R，relative_T为当前帧到参考帧（第l帧）的坐标系变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    // 4. 对窗口中每个图像帧求解sfm问题
    // 得到所有图像帧相对于参考帧l的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        // 求解失败则边缘化最早一帧并滑动窗口
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //5. 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;

    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        cv::Mat r, rvec, t, D, tmp_r;
        // 初始化估计值
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }

        // Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector; // 3D坐标
        vector<cv::Point2f> pts_2_vector; // 2D坐标
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;// 特征点索引
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                // 出现过，否则it就等于.end()了
                if(it != sfm_tracked_points.end())
                {
                    // 
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        //保证特征点数大于5
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /** 
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量
         *   OutputArray     tvec,       平移向量
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)   
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        //这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // 6.进行视觉惯性联合初始化
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate  
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;

    //1、计算陀螺仪偏置bg，尺度s，重力加速度g和速度v
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // 2、得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;// 是ROS吗？
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // 3、重新计算特征点的深度
    // 先将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));// 特征点深度estimated_depth
    // 尺度先假设
    double s = (x.tail<1>())(0);

    // 4、陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    
    // 5、将Ps、Vs、depth进行更新
    for (int i = frame_count; i >= 0; i--)
        // 5.1 Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        // 之前相机第l帧为参考系，转换到IMU bo为基准坐标系
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            //5.2 Vs为优化得到的速度
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 5.3 逆深度depth更新
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 6 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;

    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 7/所有变量从参考坐标系c0旋转到世界坐标系w
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }

    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/**
 * @brief   返回滑动窗口中第一个满足视差的帧，为l帧，以及RT,可以三角化。
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 从第一帧开始到滑动窗口中第一个满足视差足够的帧，这里的l帧之后作为参考帧做全局SFM用
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    for (int i = 0; i < WINDOW_SIZE; i++) // 10
    {
        // 寻找第i帧到窗口最后一帧的对应特征点
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);// 先得到第i帧和最后一帧的特征匹配
        if (corres.size() > 20) // 1. 首先corres数目足够
        {
            // 2. 计算平均视差>30
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());

            //判断是否满足初始化条件：视差>30和内点数满足要求
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;// i赋值给l，第l帧为参考帧
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

//三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

//vector转换成double数组，因为ceres使用数值数组
//Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    // 1.遍历滑动窗，IMU的15个自由度的优化变量
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // P、R
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
        // V、Ba、Bg
        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    // 2.Cam到IMU的外参，6自由度由7个参数表示
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }
    
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    // 3、保证传感器同步的时钟差
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    // 窗口第一帧之前的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    // 优化后的位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    // 求得优化前后的姿态差
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }
    // 根据位姿差做修正，即保证第一帧优化前后位姿不变
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

//系统故障检测 -> Paper VI-G
bool Estimator::failureDetection()
{
    //在最新帧中跟踪的特征数小于某一阈值
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }

    //偏置或外部参数估计有较大的变化
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */

    //最近两个估计器输出之间的位置或旋转有较大的不连续性
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}


/**
 * @brief   基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 * @Description 1.添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，Cam到IMU外参，时钟差
 *              2.添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *              根据倒数第二帧是不是关键帧确定边缘化的结果           
 * @return      void
*/
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;// 残差
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0); // 柯西核函数
    
    // 1.添加ceres参数块,待优化变量 AddParameterBlock
    // 1.1 Ps、Rs转变成para_Pose 6自由度7DOF（x,y,z,qx,qy,qz,w），
    //     Vs、Bas、Bgs转变成para_SpeedBias  9自由度9DOF(vx,vy,vz,bax,bay,baz,bgx,bgy,bgz)
    for (int i = 0; i < WINDOW_SIZE + 1; i++) // 遍历滑动窗数量
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    // 1.2 camera到IMU的外参也添加到估计 para_Ex_Pose
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 1.3 滑窗内第一个时刻相机到IMU时钟差，保证传感器同步， para_Td
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    //因为ceres用的是double数组，所以在下面用vector2double做类型装换
    vector2double();
    // 2. 添加残差 AddResidualBlock
    // 2.1 添加边缘化残差，丢弃帧的约束 last_marginalization_info
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // 2.2添加IMU残差 pre_integrations
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;

    // 2.3添加视觉残差 ProjectionTdFactor
    for (auto &it_per_id : f_manager.feature)// 遍历滑动窗所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;
        // 从特征点第一个观察帧开始
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());

                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    // 2.4添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }
    // 3.ceres求解Solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    // 防止优化结果在零空间变化，通过固定第一帧的位姿
    // 把ceres求解出来的结果附加到滑窗内变量中，下面开始边缘化
    double2vector();

    TicToc t_whole_marginalization;

    // 4.边缘化处理
    // 如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：  
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        //1、将上一次先验残差项传递给marginalization_info,并去除要丢弃的状态量
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // 1.1 定义损失函数 marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            // 1.2 定义残差
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            // 1.3 将各个残差以及残差涉及的优化变量添加到 marginalization_info
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2、将第0帧和第1帧间的预积分观测IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 2.1 定义损失函数IMUfactor
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                // 2.2 定义残差，优化变量为para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]，都marg掉
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                // 2.3将各个残差和优化变量添加到 marginalization_info
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        //3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        // 3.1定义代价函数
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                        // 3.2定义残差块                                                  it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                        // 3.3 将各个残差和优化变量添加到 marginalization_info                                                               vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;

    
        //4、计算每个残差对应的Jacobian，并将各参数块拷贝到统一的内存（parameter_block_data）中
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;

        //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        //6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)// 从1开始，因为第1帧状态不要
        {
            //第i的位置存放的的是i-1的内容，这就意味着窗口向前移动了一格
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;// 删除掉上一次marg相关内容
        last_marginalization_info = marginalization_info;// 更新
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }

    //如果次新帧不是关键帧：
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            //1.保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 三部曲
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            //2、premargin
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            //3、marginalize
            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //4.调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief   实现滑动窗口all_image_frame的函数
 * @Description 1.如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
                2.如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 * @return      void
*/
void Estimator::slideWindow()
{
    TicToc t_margin;
    // 边缘化最老帧
    if (marginalization_flag == MARGIN_OLD)
    {
        // 1、保存最老帧信息
        back_R0 = Rs[0];
        back_P0 = Ps[0];

        // 新来的到了滑动窗最后一帧 11
        if (frame_count == WINDOW_SIZE)
        {
            // 2、滑窗内所有信息前移1-10的变化前移
            // IMU状态量（PVQ、Ba、Bg）、IMU预积分量、线速度、角速度、dt、时间戳
            for (int i = 0; i < WINDOW_SIZE; i++)// 遍历所有
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }

            // 3.第11帧情况更新
            // 3.1、第10帧信息给了11帧（第10、11帧相同都是新来的）
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            // 3.2、新实例化一个IMU预积分给到第11帧
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            // 3.3、清空第11帧的三个buf
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            // 4、最老帧之前预积分和图像都删除
            if (true || solver_flag == INITIAL)
            {
                double t_0 = Headers[0].stamp.toSec();
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);// 最老帧

                delete it_0->second.pre_integration;// 删除预积分信息
                all_image_frame.erase(all_image_frame.begin(), it_0);// 最老帧和之前图像帧都删除
            }
            // 5、首次在原来最老帧出现的特征点转移到现在现在最老帧
            slideWindowOld();
        }
    }
    else// 次新帧不是关键帧，删除次新帧，但是不删除IMU约束
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                // 1. 
                // 取出最新一帧的信息
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                // 第5帧信息复制到第4帧上
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);

                // 留着IMU约束，当前帧和前一帧的IMU预积分转化为当前帧和前二帧
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
            }
            
            // 第5帧IMU信息复制到第4帧上
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // 更新第5帧
            // 新实例化一个IMU预积分给到第5帧
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            // 清空第5帧的三个buf
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            
            // 滑动窗
            slideWindowNew();
        }
    }
}

//滑动窗口边缘化次新帧时处理特征点被观测的帧号
//real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

//滑动窗口边缘化最老帧时处理特征点被观测的帧号
//real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    // 1、统计一共多少次merg滑窗第一帧情况
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)// 非线性
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //back_R0、back_P0为窗口中最老帧的位姿
        //Rs[0]、Ps[0]为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];// 滑窗原来第0帧
        R1 = Rs[0] * ric[0];// 滑窗原来第1帧（现在最老的第0帧）
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        // 首次在原来最老帧出现的特征点转移到现在现在最老帧
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
    // 当最新一帧是关键帧，merg滑窗内最老帧
        f_manager.removeBack();
}

/**
 * @brief   进行重定位
 * @optional    
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
*/
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

