#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

void Estimator::setParameter()  // 它读取了每一个相机到IMU坐标系的旋转/平移外参数和非线性优化的重投影误差部分的信息矩阵
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();  // 这里可以看到虚拟相机的用法
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

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
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false, // ？？？逗号是shenmegui
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
/*
对imu数据进行处理，包括更新预积分量，和提供优化状态量的初始值
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu) // first_imu == false：未获取第一帧IMU数据
    {
        first_imu = true;
        // 如果当前帧不是第一帧IMU，那么就把它看成第一个IMU，而且把他的值取出来作为初始值
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就保持不变了
    // pre_integrations[frame_count]，它是IntegrationBase的一个实例，它保存着frame_count帧中所有跟IMU预积分相关的量，包括F矩阵，Q矩阵，J矩阵等
    // 由于预积分是帧间约束，因此第一个预积分量实际上是用不到的
    if (!pre_integrations[frame_count])  // 如果当前IMU帧没有构造IntegrationBase，那就构造一个，后续会用上
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count != 0) // 在初始化时，第一帧图像特征点数据没有对应的预积分
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            // 这个量用来做初始化用的
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // 保存传感器数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 中值积分，更新滑窗中状态量，本质是给非线性优化提供可信的初始值
        int j = frame_count; 
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 传入的参数分别是当前帧上的所有特征点和当前帧的时间戳
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());

    // 把当前帧图像（frame_count）的特征点添加到f_manager.feature容器中，同时进行是否关键帧的检查
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))  // true：上一帧是关键帧，marg_old; false:上一帧不是关键帧marg_second_new
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    // 将图像数据、时间、临时预积分值存到图像帧类中
    // imageframe用于融合IMU和视觉信息的数据结构，包括了某一帧的全部信息:位姿，特征点信息，预积分信息，是否是关键帧等。
    // all_image_frame用来做初始化相关操作，他保留滑窗起始到当前的所有帧
    // 有一些帧会因为不是KF，被MARGIN_SECOND_NEW，但是即使较新的帧被margin，他也会保留在这个容器中，因为初始化要求使用所有的帧，而非只要KF
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    // 这里就是简单的把图像和预积分绑定在一起，这里预积分就是两帧之间的，滑窗中实际上是两个KF之间的
    // 实际上是准备用来初始化的相关数据
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe)); // 每读取一帧图像特征点数据，都会存入all_image_frame
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};  // 更新临时预积分初始值

    // 如果没有外参初值，则采用旋转外参初始化
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 调用getCorresponding获取frame_count和frame_count-1帧的匹配点
            // 这里标定IMU和相机的旋转外参的初值
            // 因为预积分是相邻帧的约束，因为这里得到的图像关联也是相邻的
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;  // 标志位设置成可信的外参初值
            }
        }
    }

    if (solver_flag == INITIAL) // 需要初始化
    {
        if (frame_count == WINDOW_SIZE) // 滑动窗口中塞满了才进行初始化
        {
            bool result = false;
            // 要有可信的外参值，同时距离上次初始化不成功至少相邻0.1s
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();  // 执行视觉惯性联合初始化
               initial_timestamp = header.stamp.toSec();  // 更新初始化时间戳
            }
            if(result) // 初始化操作成功，执行一次非线性优化
            {
                solver_flag = NON_LINEAR;
                // 非线性优化求解VIO
                solveOdometry();
                // 滑动窗口
                slideWindow();
                // 移除无效地图点
                f_manager.removeFailures(); // 去除滑出了滑动窗口的特征点
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];  // 滑窗中最新的位姿
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];  // 滑窗中最老的位姿
                last_P0 = Ps[0];
                
            }
            else
                slideWindow(); // 初始化不成功，对窗口进行滑动
        }
        else
            frame_count++; // 滑动窗口没塞满，接着塞
    }
    else // 已经成功初始化，进行正常的VIO紧耦合优化
    {
        TicToc t_solve;
        solveOdometry(); // 紧耦合优化的入口函数
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 失效检测，如果失效则重启VINS系统
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        // 对窗口进行滑动
        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * vins系统初始化
 * 1.确保IMU有足够的excitation
 * 2.检查当前帧（滑动窗口中的最新帧）与滑动窗口中所有图像帧之间的特征点匹配关系，
 *   选择跟当前帧中有足够多数量的特征点（30个）被跟踪，且由足够视差（2o pixels）的某一帧，利用五点法恢复相对旋转和平移量。
 *   如果找不到，则在滑动窗口中保留当前帧，然后等待新的图像帧
 * 3.sfm.construct 全局SFM 恢复滑动窗口中所有帧的位姿，以及特特征点三角化
 * 4.利用pnp恢复其他帧
 * 5.visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
 * 6.给滑动窗口中要优化的变量一个合理的初始值以便进行非线性优化
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    // 1.通过计算滑窗内所有帧的线加速度的标准差，判断IMU是否有充分运动激励，以判断是否进行初始化
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第二帧开始检查imu.第一次循环，求出滑窗内的平均线加速度
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 累加每一帧带重力加速度的deltav
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        // 第二次循环，求出滑窗内的线加速度的标准差
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);  // 求加速度的方差
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));  // 得到加速度的标准差
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }

    // global sfm
    Quaterniond Q[frame_count + 1]; // 滑动窗口中每一帧的姿态。因为滑窗的容量为10，再加上当前最新帧，所以需要存储11帧的值
    Vector3d T[frame_count + 1]; // 滑动窗口中每一帧的位置
    map<int, Vector3d> sfm_tracked_points;

    
    // 用于视觉初始化的图像特征点数据
    vector<SFMFeature> sfm_f;  // 定义一个容器来保存每个特征点的信息
    for (auto &it_per_id : f_manager.feature)  // 滑窗中出现的所有特征点
    {
        int imu_j = it_per_id.start_frame - 1;  // 这个跟imu无关，就是存储观测特征点的帧的索引
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)  // 对于当前特征点，在每一帧的坐标
        {
            imu_j++; // 观测到该特征点的图像帧的帧号
            Vector3d pts_j = it_per_frame.point;  // 当前特征点在编号为imu_j++的帧里的归一化坐标
            // 索引以及各自坐标系下的坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));  // 把当前特征在当前帧的坐标和当前帧的编号配上对
        }   // tmp_feature.observation里面存放着的一直是同一个特征，每一个pair是这个特征在 不同帧号 中的 归一化坐标
        sfm_f.push_back(tmp_feature);
    } 


    Matrix3d relative_R; // 从最新帧到选定帧(第l帧)的旋转 R_l = relative_R * R_最新帧
    Vector3d relative_T; // 从最新帧到选定帧(第l帧)的位移
    int l; // 选定帧在滑动窗口中的帧号

    // 2.在滑窗(0-9)中找到第一个满足要求的帧(第l帧)，它与最新一帧(frame_count=10)有足够的共视点和视差，并求出这两帧之间的相对位置变化关系
    // 如果找不到，则初始化失败
    if (!relativePose(relative_R, relative_T, l))
    {// 这里的第L帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧l，会作为 参考帧 到下面的全局sfm使用，得到的Rt为当前帧到第l帧的坐标系变换Rt
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }

    // 3.初始化滑动窗口中全部初始帧的相机位姿和特征点空间3D位置
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;  // 在初始化后期的第一次slidingwindow() marg掉的是old帧
        return false;
    }

    // solve pnp for all frame
    // 上面只针对KF进行sfm，初始化需要all_image_frame中的所有元素，因此下面通过KF来求解其他非KF的位姿
    // 4.对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)  // 遍历所有的图像帧
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 对于滑窗内的帧，把它们设为关键帧，并获得它们对应的IMU坐标系到l系的旋转平移
        // Headers，Q和T，它们的size都是WINDOW_SIZE+1，它们存储的信息都是滑窗内的
        if((frame_it->first) == Headers[i].stamp.toSec())  // 通过时间戳判断是不是滑窗内的帧
        {
            frame_it->second.is_key_frame = true;  // 滑动窗口中所有帧都是关键帧
            // 根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态
            // ImageFrame里不仅包括了图像信息，还包括了对应的IMU的位姿信息和IMU预积分信息，
            // 而这里，是这些帧第一次获得它们对应的IMU的位姿信息的位置，也就是bk->l帧的旋转平移
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();  // 这里对应论文公式(14)
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())  // 边界判断：如果当前帧的时间戳大于滑窗内第i帧的时间戳，那么i++，
        {
            i++;  // 第i个关键帧
        }

        // 为滑动窗口外的帧提供一个初始位姿,最近的KF提供一个初始值，Twc -> Tcw
        // 这里的 Q和 T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵，两者具有对称关系
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);  // eigen -> cv
        cv::Rodrigues(tmp_r, rvec);  // 罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false; // 初始化时位于滑动窗口外的帧是非关键帧
        vector<cv::Point3f> pts_3_vector; // 用于pnp解算的3D点
        vector<cv::Point2f> pts_2_vector; // 用于pnp解算的2D点
        for (auto &id_pts : frame_it->second.points) // 对于该帧中的特征点
        {
            int feature_id = id_pts.first; // 特征点id
            for (auto &i_p : id_pts.second) // 由于是单目，这里id_pts.second大小就是1。i_p对应着一个相机所拍图像帧的特征点信息
            {
                it = sfm_tracked_points.find(feature_id);
                // 如果it不是尾部迭代器，说明在sfm_tracked_points中找到了相应的3D点
                if(it != sfm_tracked_points.end()) // 有对应的三角化出来的3d点
                {
                    // 记录该id特征点的3D位置
                    Vector3d world_pts = it->second;  // 地图点的世界坐标
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    // 记录该id的特征点在该帧图像中的2D位置
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6) // 如果匹配到的3D点数量少于6个，则认为初始化失败
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) // pnp求解失败
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }

        // pnp求解成功
        // cv -> eigen,同时Tcw -> Twc
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // Twc -> Twi
        // 由于尺度未恢复，因此平移暂时不转到imu系
        frame_it->second.R = R_pnp * RIC[0].transpose(); // 根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态。
        frame_it->second.T = T_pnp;  // 传入的是bk->l的旋转平移
    }

    //camera与IMU对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

// visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // 调用initial_aligment.cpp中的VisualIMUAlignment函数，完成视觉SFM的结果与IMU预积分结果对齐
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 传递所有图像帧的位姿Ps、Rs,并将其设置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        // 滑动窗口中各图像帧在世界坐标系下的旋转和平移
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();  // 根据有效特征点数初始化这个动态向量
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;  // 将所有特征点的深度置为-1
    f_manager.clearDepth(dep);  // 特征点管理器把所有的特征点逆深度也设置为-1

    //triangulat on cam pose , no tic
    // 重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 多约束三角化所有的特征点，仍是尺度模糊的
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    // 将滑窗里的预积分重新计算
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);  // IMU的bias改变，重新计算滑窗内的预积分
    }

    // 下面开始把所有状态对齐到第0帧的IMU坐标系
    for (int i = frame_count; i >= 0; i--)
        // twi - tw0 = t0i，就是把所有平移对齐到滑窗中的第0帧
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // 把求解出来的KF的速度赋值给滑窗中
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // 当时求得的速度是IMU系，现在转到world系
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 把尺度模糊的3d点恢复到真实尺度下
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    Matrix3d R0 = Utility::g2R(g);  // g是枢纽帧下的重力方向，得到R_w_j
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();  // Rs[0]实际上是R_j_0
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 初始的航向为0
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;

    for (int i = 0; i <= frame_count; i++)
    {
        // 全部对齐到重力下，同时yaw角对齐到第一帧
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

/*
从最开始一帧向后扫描
首先判断与最后一帧的匹配点是不是足够多
然后判断视差是不是足够大
调用solveRelativeRT，如果内点足够多，则该帧是枢纽帧
*/
// 在滑动窗口中，寻找与最新帧有足够多数量的特征点对应关系和视差的帧，然后用5点法恢复相对位姿
// 计算滑窗内的每一帧(0-9)与最新一帧(10)之间的视差，直到找出第一个满足要求的帧，作为我们的第l帧；
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 优先从最前面开始
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);  // 寻找第i帧到窗口最后一帧(最新帧)的对应特征点归一化坐标
        // getCorresponding()的作用就是找到当前帧与最新一帧所有特征点在对应2帧下分别的归一化坐标，并配对，以供求出相对位姿时使用
        if (corres.size() > 20)  // 要求共视的特征点足够多
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));  // 取出归一化坐标的（x, y）
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();  // 计算视差.norm()返回向量的二范数（每个元素的平方和开根号）
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());  // 计算每个特征点的平均视差
            // 有足够的视差,再通过本质矩阵恢复第i帧和最后一帧之间的 R t
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)) // 判断是否满足初始化条件：视差>30
            {   // solveRelativeRT()通过基础矩阵计算最新帧到第l帧的RT（本来是第l帧到最新帧，但是函数最后对矩阵取反）,并判断内点数目是否足够
                // 一旦这一帧与当前帧视差足够大了，那么就不再继续找下去了(再找只会和当前帧的视差越来越小）
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;  // 没有满足要求的帧，初始化失败
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        // 三角化
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());

        // 滑动窗口紧耦合优化
        optimization();
    }
}

// 由于ceres的参数块都是double数组，因此这里把参数块从eigen的表示转成double
void Estimator::vector2double()
{
    // KF的位姿
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

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
    // 外参
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
    // 特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    // 传感器时间同步
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// double -> eigen 同时fix第一帧的yaw和平移，固定了四自由度的零空间
void Estimator::double2vector()
{
    // 取出优化前的第一帧的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];
    // 固定先验信息

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // 优化后的第一帧位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    // yaw角差
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    // 防止万象节死锁问题
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 保持第一帧的yaw角不变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        // 保持第一帧的位移不变
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
    // 重新设置各个特征点的逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        //回环帧在当前帧世界坐标系下的位姿
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        //回环帧在当前帧世界坐标系和他自己世界坐标系下位姿的相对变化，这个会用来矫正滑窗内世界坐标系
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        //回环帧和重定位帧相对的位姿变换
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
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

// 进行非线性优化
void Estimator::optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);  // 柯西核函数
    // 参数块1.滑窗中位姿包括位置和姿态，11帧
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 由于姿态不满足正常加法，也就是李群上没有加法，因此需要自定义加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);  // Ps、Rs转变成para_Pose,（x,y,z,qx,qy,qz,w）
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);  // Vs、Bas、Bgs转变成para_SpeedBias,(vx,vy,vz,bax,bay,baz,bgx,bgy,bgz)
        // 因为ceres传入的都是double类型的，在vector2double()里初始化的
    }
    // 参数块2.相机imu间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)  // 如果IMU-相机外参不需要标定
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);  // 这个变量固定为常数
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 参数块3.IMU-image时间同步误差
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }
    // 实际上还有地图点，其实一般的参数块不需要调用AddParameterBlock()除非要自定义运算符，增加残差块接口时会自动绑定
    TicToc t_whole, t_prepare;
    // eigen -> double
    vector2double();
    // 通过残差约束来添加残差块
    // 上一次的边缘化结果作为这一次的先验
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);  // last_marginalization_parameter_blocks指的是被边缘化留下的先验信息
    }
    // imu预积分的约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        // 超过这个约束就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;  // 统计有多少个特征用于非线性优化
    int feature_index = -1;

    // 视觉重投影约束
    // 遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)  // 2.添加各种残差----重投影残差
    {   // IMU的残差是相邻两帧，但是视觉不是（相邻两帧的误差比较小，选择相距一定的帧可能效果是一样的，而且计算量变小了）
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;  // 必须满足出现2次以上且在倒数第二帧之前出现过
 
        ++feature_index;  // 统计有效特征数量

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;  // 得到观测到该特征点的首帧
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;  // 得到首帧观测到的特征点的归一化相机坐标

        for (auto &it_per_frame : it_per_id.feature_per_frame)  // 遍历当前特征在每一帧的信息
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            // 取出另一帧的归一化相机坐标
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)  // 在有同步误差的情况下
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
            else  // 在没有同步误差的情况下
            {
                // 构造函数就是同一个特征点在不同帧的观测
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                // 约束的变量是该特征点的第一个观测帧以及其他一个观测帧，加上外参和特征点逆深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    // 回环检测相关的约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);  // 需要优化的回环帧位姿
        int retrive_feature_index = 0;
        int feature_index = -1;
        // 遍历滑窗内的每一个点
        for (auto &it_per_id : f_manager.feature)
        {
            //更新这个点在滑窗内出现的次数
            it_per_id.used_num = it_per_id.feature_per_frame.size();

            //当前点至少被观测到2次，并且首次观测不晚于倒数第二帧
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;

            //当前这个特征点至少要在重定位帧被观测到
            if(start <= relo_frame_local_index)
            {   
                //如果都扫描到滑窗里第i个特征了，回环帧上还有id序号比i小，那这些肯定在滑窗里是没有匹配点的
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                //如果回环帧某个特征点和滑窗内特征点匹配上了
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    //找到这个特征点在回环帧上的归一化坐标
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    //找到这个特征点在滑窗内首次出现的那一帧上的归一化坐标
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    //构造重投影的损失函数
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    //构造这两帧之间的残差块
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    //回环帧上的这个特征点被用上了，继续找下一个特征点
                    retrive_feature_index++;
                }     
            }
        }

    }

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
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));  // static_cast<int>强制类型转换为int
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector();

//——————————————————————————————————————————————————————marg_old————————————————————————————————————————————————————————————————————
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        // 一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        // 先验误差会一直保存，而不是只使用一次
        // 如果上一次边缘化的信息存在
        // 要边缘化的参数块是 para_Pose[0] para_SpeedBias[0] 以及 para_Feature[feature_index](滑窗内的第feature_index个点的逆深度)

        // 上一次边缘化的结果
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {   // 查询last_marginalization_parameter_blocks中是首帧状态量的序号
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,  // 添加上一次边缘化的参数块
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);  // 这一段代码用drop_set把最老帧的先验信息干掉了
        }
        // 只有第1个预积分和待边缘化参数块相连
        {
            if (pre_integrations[1]->sum_dt < 10.0)  // 把这次要marg的IMU信息加进来
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});  // 这里就是第0和1个参数块是需要被边缘化的(这里就是最老帧)
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // 把这次要marg的视觉信息加进来，遍历视觉重投影的约束
        {   // 添加视觉的先验，只添加起始帧是旧帧且观测次数大于2的Features
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();  // 特征被观测到的次数
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))  // Feature的观测次数不小于2次，且起始帧不属于最后两帧
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;  // imu_i表示被观测到时的起始帧，imu_j表示起始帧的前一帧
                // 只找能被第0帧看到的特征点
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                // 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;  // 当前帧的特征点坐标
                    if (ESTIMATE_TD)  // 如果需要估计时延
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});  // 这里第0帧和地图点被margin
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else  // 如果不需要估计时延
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
        // 所有的残差块都收集好了
        TicToc t_pre_margin;
        // 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        // 边缘化操作
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
        // 即将滑窗，因此记录新地址对应的老地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和延时不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
//——————————————————————————————————————————————————————marg_new————————————————————————————————————————————————————————————————————
    else  // 如果第二最新帧不是关键帧的话，则把这帧的视觉测量舍弃掉（边缘化）而保留IMU测量值在滑动窗口中
    {
        if (last_marginalization_info && //判断 last_marginalization_info 是否存在，以及最后一个 Pose 参数块 是否在上一次边缘化的参数块集合中出现
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

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
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
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

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)  // 依次把滑窗内信息前移
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
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
            // 滑动窗口中最新帧的时间戳、位姿和bias，用滑动窗口中第2最新帧的数据来初始化
            // 把滑窗末尾(10帧)信息给最新一帧(11帧)把滑窗末尾(10帧)信息给最新一帧(11帧)
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};  // 新实例化一个IMU预积分对象给下一个最新一帧

            dt_buf[WINDOW_SIZE].clear();  // 清空第11帧的buf
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)  // 删除最老帧对应的全部信息
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)  // 把滑窗内最老一帧以前的帧的预积分信息全删掉
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);

            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                // 取出最新一帧的信息
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                // 当前帧和前一帧之间的 IMU 预积分转换为当前帧和前二帧之间的 IMU 预积分
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            //  用最新一帧的信息覆盖上一帧信息
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // 因为已经把第11帧的信息覆盖了第10帧，所以现在把第11帧清除
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;  // 统计一共有多少次merge滑窗第10帧的情况
    f_manager.removeFront(frame_count);  // 当最新一帧(11)不是关键帧时，用于merge滑窗内最新帧(10)
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;  // 统计一共有多少次merge滑窗第一帧的情况

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];  // 滑窗原先的最老帧(被merge掉)的旋转(c->w)
        R1 = Rs[0] * ric[0];  // 滑窗原先第二老的帧(现在是最老帧)的旋转(c->w)
        P0 = back_P0 + back_R0 * tic[0];  // 滑窗原先的最老帧(被merge掉)的平移(c->w)
        P1 = Ps[0] + Rs[0] * tic[0];  // 滑窗原先第二老的帧(现在是最老帧)的平移(c->w)
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);  // 把首次在原先最老帧出现的特征点转移到原先第二老帧的相机坐标里(仅在slideWindowOld()出现过)
    }
    else
        f_manager.removeBack();  // 当最新一帧是关键帧时，用于merge滑窗内最老帧
}

// 关于重定位
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp; //pose_graph里要被重定位的关键帧的时间戳
    relo_frame_index = _frame_index; //回环帧在pose_graph里的id
    match_points.clear();
    match_points = _match_points;    //两帧共同的特征点在回环帧上的归一化坐标
    prev_relo_t = _relo_t;           //回环帧在自己世界坐标系的位姿(准)
    prev_relo_r = _relo_r;           //回环帧在自己世界坐标系的位姿(准)
    for(int i = 0; i < WINDOW_SIZE; i++) 
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())//如果pose_graph里要被重定位的关键帧还在滑窗里
        {
            relo_frame_local_index = i; //找到它的id
            relocalization_info = 1;    //告诉后端要重定位
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j]; //用这个关键帧的位姿(当前滑窗的世界坐标系下，不准)初始化回环帧的位姿
        }
    }
}

