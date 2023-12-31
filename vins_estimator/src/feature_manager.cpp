#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

// 得到有效的地图点数目
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * 把当前帧图像（frame_count）的特征点添加到feature容器中
 * 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧）
 * 也就是说当前帧图像特征点存入feature中后，并不会立即判断是否将当前帧添加为新的关键帧，而是去判断当前帧的前一帧（第2最新帧）。
 * 当前帧图像要在下一次接收到图像时进行判断（那个时候，当前帧已经变成了第2最新帧）
 * 这部分里写了2个判断关键帧的判断指标：
 * 第一个是“the average parallax apart from the previous keyframe”，对应着代码中parallax_num和parallax_sum / parallax_num；
 * 第二个是“If the number of tracked features goes below a certain threshold, we treat this frame as a new keyframe”，对应着代码里的last_track_num。
 * 注意，这部分里还有一个函数是compensatedParallax2()，用来计算当前特征点的视差
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)  // 关键帧判断
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的总视差
    int parallax_num = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的数量
    last_track_num = 0; // 当前帧（第1最新帧）图像跟踪到的特征点的数量

    // 把当前帧图像特征点数据image添加到feature容器中
    // feature容器按照特征点id组织特征点数据，对于每个id的特征点，记录它被滑动窗口中哪些图像帧观测到了
    // 遍历每个特征点
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);  // 用特征点信息构造一个对象

        int feature_id = id_pts.first;

        /**
         * STL find_if的用法：
         * find_if (begin, end, func)
         * 就是从begin开始 ，到end为止，返回第一个让 func这个函数返回true的iterator
         */
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;  // 在已有的id中寻找是否是有相同的特征点
                          });

        // 返回尾部迭代器，说明该特征点第一次出现（在当前帧中新检测的特征点），需要在feature中新建一个FeaturePerId对象
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;  // 追踪到上一帧的特征点数目
        }
    }

    // 前两帧都设置为KF，追踪过少也认为是KF
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 计算第2最新帧和第3最新帧之间跟踪到的特征点的平均视差
    for (auto &it_per_id : feature)
    {
        // 计算的实际上是frame_count-1,也就是前一帧是否为关键帧
        // 因此起始帧至少得是frame_count - 2,同时至少覆盖到frame_count - 1帧
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 对于给定id的特征点，计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
            //（需要使用IMU数据补偿由于旋转造成的视差）
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        // 如果第2最新帧和第3最新帧之间跟踪到的特征点的数量为0，则把第2最新帧添加为关键帧
        // ？？怎么会出现这种情况？？？？
        // 如果出现这种情况，那么第2最新帧和第3最新帧之间的视觉约束关系不就没有了？？？
        return true;
    }
    else
    {
        // 计算平均视差
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX; // 如果平均视差大于设定的阈值，则把第2最新帧当作关键帧
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 得到同时被frame_count_l和frame_count_r帧看到的特征点和各自的坐标
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)  // 保证需要的特征点被这两帧都观察到
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;  // 获得在feature_per_frame中的索引
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));  // 返回相机坐标系下的坐标对
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

// 移除一些不能被三角化的点
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

// 把给定的深度赋值给各个特征点作为逆深度
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

// 得到特征点的逆深度
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 利用观测到该特征点的所有位姿来三角化特征点
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature) // 对于每个id的特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 每个id的特征点被多少帧图像观测到了
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            // 如果该特征点被两帧及两帧以上的图像观测到，
            // 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第4最新关键帧
            // 也就是说，至少是第4最新关键帧和第3最新关键帧观测到了该特征点（第2最新帧似乎是紧耦合优化的最新帧）
            continue;

        if (it_per_id.estimated_depth > 0) // 该id的特征点深度值大于0，该值在初始化时为-1，如果大于0，说明该点被三角化过
            continue;

        // imu_i：观测到该特征点的第一帧图像在滑动窗口中的帧号
        // imu_j：观测到该特征点的最后一帧图像在滑动窗口中的帧号
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        // Twi -> Twc, 第一个观察到这个特征点的KF的位姿
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 遍历所有观测到这个特征点的KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; // 观测到该特征点的最后一帧图像在滑动窗口中的帧号
            // 得到该KF的相机坐标系位姿
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            // T_w_cj -> T_c0_cj
            // 这里是将原本的cj作为世界坐标系改为c0帧作为世界坐标系
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            // T_c0_cj -> T_cj_c0相当于把c0当作世界系
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // 构建超定方程的其中两个方程
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        // 求解齐次坐标下的深度
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        // 得到的深度值实际上就是第一个观察到这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1) // 如果估计出来的深度小于0.1，则把它替换为一个设定的值
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

/*
把被移除帧看见的地图点的管理权交给当前的最老帧
marg_R、marg_P:被移除的位姿
new_R、new_P：转移地图点的位姿
*/
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果不是被移除帧看到的，那么该地图点对应的起始帧id减1
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  // 取出归一化相机坐标坐标系
            it->feature_per_frame.erase(it->feature_per_frame.begin());  // 该点不再被原来的第一帧看到，因此从中移除
            // 如果这个地图点只被一帧看到，就没有存在的价值了
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else  // 进行管辖权交接
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;  // 实际相机坐标系下的坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;  // 转到世界坐标系下
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);  // 转到新的最老帧的相机坐标系下
                double dep_j = pts_j(2);
                // 看看深度是否有效
                if (dep_j > 0)
                    it->estimated_depth = dep_j;  // 有效的话就得到现在最老帧下的深度值
                else
                    it->estimated_depth = INIT_DEPTH;  // 无效就设置默认值
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

// 这个还没有初始化结束，因此相比刚才，不进行地图点新的深度的换算，因为此时还要进行视觉惯性对齐
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 对margin倒数第二帧进行处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)  // 如果地图点被最后一帧看到，由于滑窗，它的起始帧减1
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;  // 如果这个地图点能够被倒数第二帧看到，那么倒数第二帧在这个地图点对应滑窗的索引
            if (it->endFrame() < frame_count - 1)  // 如果该地图点不能被倒数第二帧看到，那没什么好做的
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);  // 被倒数第二帧看到，erase这个索引
            if (it->feature_per_frame.size() == 0)  // 如果这个地图点没有别的观测了，就没有存在的价值了
                feature.erase(it);
        }
    }
}

/**
 * 对于给定id的特征点
 * 计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
 * （需要使用IMU数据补偿由于旋转造成的视差）
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 第3最新帧
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 第2最新帧

    double ans = 0;  // 初始化视差

    // 以下的操作暂时没有看懂
    Vector3d p_j = frame_j.point;  // 归一化相机坐标

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;  // 归一化相机坐标系的坐标差

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}