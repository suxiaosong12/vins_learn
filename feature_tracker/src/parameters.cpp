#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;  // 判断是否发布特征点

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))  // Ros自带的getParam获取配置文件地址
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    config_file = readParam<std::string>(n, "config_file");  // 从launch文件获取config_file地址
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);  // FileStorage::READ 打开文件进行读操作，读取配置文件
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");  // 用readParam函数从launch读取vins_folder地址

    fsSettings["image_topic"] >> IMAGE_TOPIC;  // 返回image_topic的节点数据给IMAGE_TOPIC
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];  // 单帧图像最大特征点数目
    MIN_DIST = fsSettings["min_dist"];  // 两个特征点之间最短的像素距离
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];  // 对极约束ransac求解的inlier门限值
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];  // 是否做均衡化处理
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
