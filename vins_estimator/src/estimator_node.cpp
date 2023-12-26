#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf; // 订阅pose_graph_node发布的回环帧数据，存到relo_buf队列中，供重定位使用
int sum_of_wait = 0;

std::mutex m_buf; // 用于处理多个线程使用imu_buf和feature_buf的冲突
std::mutex m_state; // 用于处理多个线程使用当前里程计信息（即tmp_P、tmp_Q、tmp_V）的冲突
std::mutex i_buf;
std::mutex m_estimator; // 用于处理多个线程使用VINS系统对象（即Estimator类的实例estimator）的冲突

double latest_time; // 最近一次里程计信息对应的IMU时间戳
// 当前里程计信息
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;

// 当前里程计信息对应的IMU bias
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;

// 当前里程计信息对应的IMU测量值
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

bool init_feature = 0;
bool init_imu = 1; // true：第一次接收IMU数据
double last_imu_t = 0; // 最近一帧IMU数据的时间戳


// 根据当前imu数据预测当前位姿
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) // 如果是第一帧imu，只保留上一帧的时间
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;  // 计算两帧之间的时间差
    latest_time = t;

    // 得到加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};  // 加速度

    // 得到角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};  // 角速度

    // 上一时刻世界坐标系下加速度值
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;
    // 中值陀螺仪的结果
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新姿态，通过deltaQ转换为四元数
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    // 当前时刻世界坐标系下的加速度值
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    // 加速度中值积分的值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 经典物理中位置、速度更新方程
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 用最新VIO结果更新最新IMU对应的位姿
void update()
{
    TicToc t_predict;
    latest_time = current_time;

    // 首先从估计器中得到滑动窗口中最后一个图像帧的imu更新项[P,Q,V,ba,bg,a,g]
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    // 滑动窗口中最新帧并不是当前帧，中间隔着缓存队列的数据，所以还需要使用缓存队列中的IMU数据进行积分得到当前帧的里程计信息
    // 因为imu的频率比图像频率要高很多，在getMeasurements(）将图像和imu时间对齐后，imu_buf中还会存在imu数据
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;  // 遗留的imu buffer，因为下面需要pop，所以copy一份
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

// 把图像帧和对应的IMU数据匹配好，然后获取匹配好的图像IMU组
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    // 直到把imu_buf或者feature_buf中的数据全部取出，才会退出while循环
    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // imu_buf队尾元素的时间戳，早于等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要等待接收IMU数据
        // imu      *****
        // image           ****
        // imu还没来
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu_buf队首元素的时间戳，晚于等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要剔除feature_buf队首多余的特征点数据
        // imu      ********
        // image  **********
        // 这种只能扔掉一些image帧
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        // 此时保证了图像前一定有imu数据
        // imu  **********
        // image    *   *
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front(); // 读取feature_buf队首的数据
        feature_buf.pop(); // 剔除feature_buf队首的数据
        
        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        // 一帧图像特征点数据，对应多帧imu数据,把它们进行对应，然后塞入measurements
        // 一帧图像特征点数据，与它和上一帧图像特征点数据之间的时间间隔内所有的IMU数据，以及时间戳晚于当前帧图像的第一帧IMU数据对应
        // 如下图所示：
        //   *             *             *             *             *            （IMU数据）
        //                                                      |                 （图像特征点数据）
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front()); // 时间戳晚于当前帧图像的第一帧IMU数据

        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/*
imu消息存进buffer，同时按照imu频率 预测 位姿并发送，这样可以提高里程计频率
*/
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec(); 

    // 进行线程锁的设置，将IMU数据存入IMU数据缓存队列imu_buf
    // 线程锁和条件变量参考：https://www.jianshu.com/p/c1dfa1d40f53
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒getMeasurements()读取缓存imu_buf和feature_buf中的观测数据

    // 通过IMU测量值积分更新并发布里程计信息
    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // 只有初始化完成之后才发送当前结果
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // 发布里程计信息，发布频率很高（与IMU数据同频），每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息。
            // 还有一个pubOdometry()函数，似乎也是发布里程计信息，但是它是在estimator每次处理完一帧图像特征点数据后才发布的，有延迟，而且频率也不高（至多与图像同频）
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    // 将图像特征点数据存入图像特征点数据缓存队列feature_buf
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒getMeasurements()读取缓存imu_buf和feature_buf中的观测数据
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)  // 清空buffer，优化器复位，标志位清零
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
// process()是measurement_process线程的线程函数，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        // 因为IMU的频率比视觉帧的发布频率要高，所以说在这里，需要把一个视觉帧和之前的一串IMU帧的数据配对起来
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
                    return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();  // 从缓存队列中读取数据完成，数据buffer的锁解锁，回调可以继续塞数据了

        m_estimator.lock();  // 进行后端求解，不能和复位重启冲突
        // 遍历measurements中的每一个measurement (IMUs,IMG)组合
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

            //遍历该组中的各帧imu数据，进行预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec(); // 最新IMU数据的时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 图像特征点数据的时间戳，补偿了通过优化得到的一个时间偏移
                if (t <= img_t) // 补偿时间偏移后，图像特征点数据的时间戳晚于或等于IMU数据的时间戳
                { 
                    if (current_time < 0) // 第一次接收IMU数据时会出现这种情况
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t; // 更新最近一次接收的IMU数据的时间戳

                    // IMU数据测量值
                    // 3轴加速度测量值
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;

                    // 3轴角速度测量值
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // 这里干了2件事，IMU粗略地预积分，然后把值传给一个新建的IntegrationBase对象
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));  // 进行IMU预积分
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                // 针对最后一个imu数据，需要利用倒数第二帧，图像帧和最后一帧进行插值，再调用processIMU函数
                else // 最后一帧IMU数据，时间戳晚于图像特征点数据（也是一组measurement中的唯一一帧），对IMU数据进行插值，得到图像帧时间戳对应的IMU数据
                {
                    // 时间戳的对应关系如下图所示：
                    //                                            current_time         t
                    // *               *               *               *               *     （IMU数据）
                    //                                                          |            （图像特征点数据）
                    //                                                        img_t
                    double dt_1 = img_t - current_time;  // current_time < img_time < t
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 以下操作为线性插值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 设置重定位用的回环帧
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            // 在relo_buf中取出最后一个重定位帧，拿出其中的信息并执行setReloFrame()
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)  // 有效回环信息
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec(); // 回环帧的时间戳
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // 回环帧的位姿
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r); // 设置回环帧
            }


            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());
            TicToc t_s;

            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            /*
            1、虽然它叫image，但是这个容器里面存放的信息是每一个特征点的！
            2、索引值是feature_id；
            3、value值是一个vector，如果系统是多目的，那么同一个特征点在不同摄像头下会有不同的观测信息，
                那么这个vector，就是存储着某个特征点在所有摄像头上的信息。对于VINS-mono来说，value它不是vector，仅仅是一个pair
            4、接下来看这个vector里面的每一pair。int对应的是camera_id，告诉我们这些数据是当前特征点在哪个摄像头上获得的。
            5、Matrix<double, 7, 1>是一个7维向量，依次存放着当前feature_id的特征点在camera_id的相机中的归一化坐标，像素坐标和像素运动速度
            */

            // 遍历img_msg里面的每一个特征点的归一化坐标
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {   // 把img的信息提取出来放在image容器里去，通过这里，可以理解img信息里面装的都是些什么
                int v = img_msg->channels[0].values[i] + 0.5; // channels[0].values[i]==id_of_point
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;  // 相机系坐标
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];  // 特征点像素坐标
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];  // 特征点速度
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);  // 检查是不是归一化
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);  // 处理图像帧：初始化，紧耦合的非线性优化

            // 一些打印以及topic的发送
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            // 向RVIZ发布里程计信息、关键位姿、相机位姿、点云和TF关系，这些函数都定义在中utility/visualization.cpp里，都是ROS相关代码
            // 每处理完一帧图像特征点数据，都要发布这些话题
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // VINS系统完成滑动窗口优化后，用优化后的结果，更新里程计数据
            update();  // 更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // RViz相关话题
    registerPub(n);  // 注册visualization.cpp中创建的发布器
    
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());  // 接受imu消息
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);  // 接受前端视觉光流结果
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);  // 接受前端重启命令，判断是否重启estimator
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);  // 回环检测的fast relocalization响应，根据回环检测信息进行重定位

    // estimator_node中的线程由旧版本中的3个改为现在的1个
    // 回环检测和全局位姿图优化在新增的一个ROS节点中运行
    // measurement_process线程的线程函数是process()，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
    std::thread measurement_process{process};  // 调用核心处理线程处理push到buffer中的数据
    ros::spin();

    return 0;
}
