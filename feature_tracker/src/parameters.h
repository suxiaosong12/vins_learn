#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;  // 图片高度
extern int COL;  // 图片长度
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;  //单目相机


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;  // 发布特征点的频率(也就是bk/ck帧与bk+1/ck+1帧之间的频率)，注意，FREQ要比实际的照片频率要慢
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;

void readParameters(ros::NodeHandle &n);
