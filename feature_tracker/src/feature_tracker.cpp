#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)  // 判断点是否在图片内
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);  // 四舍五如返回整形
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)  // 根据经过处理的点的状态对点进行删减，若该点状态值为0则舍去
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);  // 调整容器大小,销毁多余元素
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()  // 给现有的特征点设置mask，目的为了特征点的均匀化
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));  // 将图像设置成单一灰度和颜色white
    
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;  // 构造(track_cnt，forw_pts，ids)序列

    for (unsigned int i = 0; i < forw_pts.size(); i++)  //让他们在排序的过程中仍然能一一对应
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    // 使用 lambda function 匿名函数排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {  // 对光流跟踪到的特征点forw_pts，按照被跟踪到的次数cnt从大到小排序
            return a.first > b.first;  // 降序排列
         });

    // 清空track_cnt，forw_pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)  // 遍历cnt_pts_id，使图像提取的特征点更均匀
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {   // 当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);  // 图片，点，半径，颜色为0,粗细（-1）表示填充
        }  // 在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点（使跟踪点不集中在一个区域上）
    }
}

void FeatureTracker::addPoints()  // 将新的特征点加入容器中，id设置为-1进行区分
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

/*
_img：输入图像，_cur_time：图像的时间戳
1、图像均衡化处理
2、光流追踪
3、提取新的特征点（如果发布）
4、所有特征点去畸变，计算速度
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)  // 如果图像过亮或者过暗，进行直方图均衡化，提升对比度，方便提取角点
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));  // 对图像进行自适应直方图均衡化
        TicToc t_c;
        clahe->apply(_img, img);  // apply(InputArray src, OutputArray dst)
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 首帧判断和对forw_img更新
    if (forw_img.empty())  // 如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();  // 光流追踪和失败点剔除

    if (cur_pts.size() > 0)  //上一帧有特征点，就可以进行光流追踪了
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 调用opencv函数进行光流追踪
        // Step 1 通过opencv光流追踪给的状态位剔除outlier
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        // calcOpticalFlowPyrLK() OpenCV的光流追踪函数,提供前后两张图片以及对应的特征点，即可得到追踪后的点，以及各点的状态、误差
        // Step 2 通过图像边界剔除outlier
        for (int i = 0; i < int(forw_pts.size()); i++)  // 将位于图像边界外的点标记为0
            if (status[i] && !inBorder(forw_pts[i]))  // 剔除状态为1，但位于图像边界外的特征点
                status[i] = 0;
        //根据status,把跟踪失败的点剔除
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除  
        reduceVector(prev_pts, status);  // 没用到
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);  // 特征点的id
        reduceVector(cur_un_pts, status);  // 去畸变后的坐标
        reduceVector(track_cnt, status);   // 追踪次数
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    // 更新当前特征点被追踪到的次数
    for (auto &n : track_cnt)  // 光流追踪成功,特征点被成功跟踪的次数就加1
        n++;                   // 数值代表被追踪的次数，数值越大，说明被追踪的就越久

    // 通过基本矩阵剔除外点
    if (PUB_THIS_FRAME)  // 如果需要将这一帧发送给后端
    {
        rejectWithF();  // 利用对极约束进行特征点筛选
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();  // 对跟踪点进行排序并去除密集点
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        // 计算需要多少提取多少特征点，一共需要MAX_CNT个特征点，当前有static_cast(forw_pts.size())个特征点，需要补充n_max_cnt个特征点
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);  // 提取新特征点
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    // 将当前帧的信息保存到上一帧，去畸变，计算特征点速度
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;  // 以上三个量无用
    cur_img = forw_img;  // 上一帧的图像
    cur_pts = forw_pts;  // 上一帧的特征点
    undistortedPoints();
    // 这个函数干了2件事，第一个是获取forw时刻去畸变的归一化坐标(这个是要发布到rosmsg里的points数据)，另一个是获取forw时刻像素运动速度
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()  // 通过基本矩阵去除外点
{
    if (forw_pts.size() >= 8)  // 当前被追踪到的光流至少8个点
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        // 先把特征点坐标(像素坐标)转为归一化坐标（去畸变过程），再转回到像素坐标，然后再用findFundamentalMat()找outlier
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());  // 分别是上一帧和当前帧去畸变的像素坐标
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;  
            // 得到相机归一化坐标系的值
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 利用liftProjective函数将像素坐标转化为无畸变的归一化坐标,
            // 返回的是去畸变后的归一化坐标,投影回像素坐标系时，使用的内参焦距固定为460，cx和cy固定为图像宽度和高度的一半
            // 这种策略可以使像素阈值固定下来（VINS-Mono）中始终为3，不用为每个相机模型分别设置阈值
            // 这里用一个虚拟相机，https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/48
            // 这里有个好处就是对F_THRESHOLD和相机无关,投影到虚拟相机的像素坐标系
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;  // 转换为像素坐标
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);  // 去畸变
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;  // opencv接口计算基本矩阵
        // 目的并不是为了得到本质矩阵，而是通过计算本质矩阵来剔除外点
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;  // means all the pts are identified
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)  // 读取相机内参
{   // 读到的相机内参赋给m_camera
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    // generateCameraFromYamlFile通过配置文件读取相机的模型（一般为针孔相机模型，根据不同相机类型生成不同camera对象
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()  // 当前帧所有点统一去畸变，同时计算特征点速度，用来后续时间戳标定
{   
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);  // 有的之前去过畸变了，这里连同新加入的特征点重新做一次
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())  // 判断上一帧中特征点点是否为0，不为0则计算点跟踪的速度
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);  // 找到同一个特征点
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));  // 得到在归一化平面的速度
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
