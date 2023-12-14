#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];  // 存储当前时刻所有相关的数据
double first_image_time;  // 第一帧图像时间戳
int pub_count = 1;  // 每隔delta_t = 1/FREQ 时间内连续(没有中断/没有报错)发布的帧数
bool first_image_flag = true;  // 1:当前是第一帧  0:当前不是第一帧
double last_image_time = 0;  // 上一帧图像的时间戳
bool init_pub = 0;  // 0:第一帧不把特征发布到buf里    1:发布到buf里


// 该函数是ROS的回调函数，主要功能包括：readImage()函数对新来的图像使用光流法进行特征点跟踪，
// 并将追踪的特征点封装成feature_points发布到pub_img的话题下，将图像封装成ptr发布在pub_match下
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)  // 接收图像
{
    if(first_image_flag)  // 对于第一帧图像，只记录对应时间戳，不提取特征，因为他没有前一帧图像，无法获取光流
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();  // get the time of image
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    // 对于时间戳错乱的帧(时间戳间隔超过1s，图像时间差太多光流追踪就会失败)，重新初始化
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;  // 0:不重启  1：重启
        restart_flag.data = true;
        pub_restart.publish(restart_flag);  // 告诉其他模块要重启了
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();  // 如果时间戳没问题，记录当前时间

    // frequency control，控制一下发给后端的频率在10Hz左右
    // 计算当前累计的pub_count个帧的频率与FREQ的关系
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {// 要想发布图像帧，那么实际频率要比设定值小;但是，如果实际频率与设定频率的累积误差大于0.01了，就不能发布这一帧
        PUB_THIS_FRAME = true;  // 发布特征点
        // 如果这段时间的频率和预设频率十分接近，就认为这段时间很棒，重启一下，避免delta_t太大
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {// this "if" make sure the frequece is not so high
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;  // 如果实际发布频率大于设定值，肯定就不发了
    
    // 图像的格式调整和图像读取
    // 读取sensor_msgs::Image img的数据，并转为MONO8格式，用cv::Mat show_img接收
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")  // 8位灰度图像 
    {
        sensor_msgs::Image img;  // Change 'sensor_msgs::ImageConstPtr'(const) to non-const
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
        // 将图像编码8UC1转换为mono8,即存储下来的图像为单色，8Bit的图片，一般是bmp，jpeg等
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    // img_msg和img都是sensor_msg格式的，我们需要一个桥梁，转换为CV::Mat格式的数据，以供后续图像处理
    cv::Mat show_img = ptr->image;

    TicToc t_r;
    
    //对最新帧forw的特征点的提取和光流追踪(该函数核心部分！！！)；基本上调用了feature_tracker.cpp里面的全部函数
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)  // 单目时：FeatureTracker::readImage() 函数读取图像数据进行处理
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
            // readImage()传了2个参数，当前帧的图像和当前帧的时间戳
            // readImage()函数作用：1.图像均衡化预处理；2.光流追踪；3提取新的特征点（如果发布；4所有特征点去畸变，计算速度
        else
        {
            if (EQUALIZE)  // 判断是否对图像进行自适应直方图均衡化
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)  // 更新全局ID
    {
        bool completed = false;  // completed如果是true，说明没有更新完id，则持续循环，如果是false，说明更新完了则跳出循环
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);  // |=按位或并赋值，单目相机可直接取=
        if (!completed)
            break;
    }

   // 特征点的发布
   if (PUB_THIS_FRAME)  // 如果PUB_THIS_FRAME=1则进行发布
   {
        pub_count++;  // 更新计数器
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);  // 归一化坐标
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;  // 像素坐标x
        sensor_msgs::ChannelFloat32 v_of_point;  // 像素坐标y
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;  // 去畸变的归一化坐标
            auto &cur_pts = trackerData[i].cur_pts;  // 像素坐标
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;  // 归一化坐标下的速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)  // 只发布追踪次数大于1的特征点
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;  // 归一化坐标
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    // 利用这个ros消息的格式进行信息存储
                    feature_points->points.push_back(p);  // 归一化坐标
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);  // 像素坐标
                    v_of_point.values.push_back(cur_pts[j].y);  // 像素坐标
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }  // 将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)，
        }      // 封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image
        // 如果是第一帧的话，不发布数据
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);  // 前端得到的信息通过这个publisher发布出去

        // 可视化相关操作
        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2); // 特征点被追踪的次数越多越红，越少越蓝
                    
                    //draw speed line
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");  // ros节点初始化
    ros::NodeHandle n("~");  // 声明一个句柄，~代表这个节点的命名空间，"~"参数代表节点的命名空间在feature_tracker下
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);  // 设置ros log级别
    readParameters(n);  // 读取参数，config->euroc->euroc_config.yaml中的一些配置参数

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);  // 读取每个相机内参，只有一个相机

    if(FISHEYE)  // 判断是否加入鱼眼mask来去除边缘噪声
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // 订阅话题IMAGE_TOPIC（/cam0/image_raw），有图像发布到这个话题上的时候，执行回调函数，这里直接进入到img_callback函数中接收图像
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);  // 缓存大小为100个消息，每收到一次消息，运行回调函数

    // 分别在feature，feature_img，restart话题上发布消息
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);  // 跟踪的特征点图像，主要是之后给RVIZ用和调试用
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);  // 即跟踪的特征点信息，由/vins_estimator订阅并进行优化 
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);  // 判断特征跟踪模块是否出错，若有问题则进行复位，由/vins_estimator订阅
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();  // ROS消息回调处理函数,没有的话回调函数将永远不会调用
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?