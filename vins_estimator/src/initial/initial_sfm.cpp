#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

// 三角化恢复特征点3D位置
// 参考 https://blog.csdn.net/jsf921942722/article/details/95511167
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	//|-r1 + v * r2| 
	//| r0 - u * r2|* X = 0, 一个归一化平面坐标x和变换矩阵可以构建两个关于X的线性方程组
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);  // u*r2-r0
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);  // v*r2-r1
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // SVD分解
	point_3d(0) = triangulated_point(0) / triangulated_point(3);  // 归一化
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// PNP求解位姿
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector; // 用于pnp解算的2D点
	vector<cv::Point3f> pts_3_vector; // 用于pnp解算的3D点
	for (int j = 0; j < feature_num; j++) // 遍历sfm_f中的特征点
	{
		if (sfm_f[j].state != true) // 如果该特征点未被三角化，则跳过该特征点
			continue;
		Vector2d point2d;
		// 找出代求帧上所有特征点的归一化坐标和3D坐标（l系上）
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // 遍历观测到该特征点的图像帧
		{
			if (sfm_f[j].observation[k].first == i) // 从observation中找到待解算位姿的帧i
			{
				// 获取用于pnp解算的2D点
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);

				// 获取用于pnp解算的3D点
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break; // 因为一个特征在帧i上只会出现一次，一旦找到了就没有必要再继续找了
			}
		}
	}
	if (int(pts_2_vector.size()) < 15) // 如果内点数量小于15，则表明初始化过程中特征点跟踪不稳定
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10) // 如果内点数量小于10，则认为无法完成pnp解算
			return false;
	}
	//套用opencv公式求解pnp
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);  // 转换为solvePnP能处理的格式
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1); // 似乎是因为2D点使用了归一化平面上的坐标，所以相机内参矩阵设置为单位阵
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);  // 得到了第l帧到第i帧的旋转平移
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r); // 旋转向量转为旋转矩阵
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);  //转换为原有格式
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp; // 输出旋转矩阵
	P_initial = T_pnp; // 输出平移向量
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++) // 遍历sfm_f中的特征点
	{
		if (sfm_f[j].state == true) // 如果该特征点已经被三角化，则跳过
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0; // frame0中的2D点
		Vector2d point1; // frame1中的2D点
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // 遍历观测到该特征点的图像帧
		{
			if (sfm_f[j].observation[k].first == frame0) 
			{
				// frame0观测到了该特征点，记录该特征点的坐标
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				// frame1观测到了该特征点，记录该点的坐标
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1) // frame0和frame1同时观测到了该特征点，则进行三角化
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true; // 该特征点被成功三角化
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

/**
 * 输入参数说明：
 * int frame_num： 滑动窗口中图像帧的数量，实际上就是WINDOW_SIZE + 1
 * Quaterniond* q： 一个数组指针，数组中需要存储滑动窗口中所有帧的姿态
 * Vector3d* T： 一个数组指针，数组中需要存储滑动窗口中所有帧的位置
 * int l： 在滑动窗口中找到的与最新帧做5点法的帧的帧号
 * const Matrix3d relative_R： 从最新帧到第l帧的旋转
 * const Vector3d relative_T： 从最新帧到第l帧的位移
 * vector<SFMFeature> &sfm_f： 用于视觉初始化的特征点数据
 * map<int, Vector3d> &sfm_tracked_points
 */
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;

	// have relative_r relative_t
	// intial two view
	// 把relativePose()找到的第l帧（l是在滑动窗口中找到的与最新帧做5点法的帧的帧号）作为初始位置
	// 第l帧的姿态设置为一个没有任何旋转的实单位四元数
	// 枢纽帧设置为单位帧，也可以理解为世界系原点
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;

	// 第l帧位置向量设置为[0, 0, 0],作为参考帧
	T[l].setZero();

	// 这里把第l帧看作参考坐标系，根据当前帧相对于第l帧的relative_R，relative_T，
	// 得到当前帧在参考坐标系下的位姿，之后的pose[i]表示第l帧相对于第i帧的变换矩阵[R|T]
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); // relative_R表示最新帧相对于l帧的旋转矩阵Rln,(Xl = Rln * Xn)
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	// 构造容器，存储滑窗内 第l帧 相对于 其它帧 和 最新一帧 的变换矩阵
	// 这些容器存储的都是相对运动，大写的容器对应的是l帧旋转到各个帧。
    // 小写的容器是用于全局BA时使用的，也同样是l帧旋转到各个帧。
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	//for ceres
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];

	Eigen::Matrix<double, 3, 4> Pose[frame_num]; // 滑动窗口中第l帧相对于第i帧的变换矩阵[R t]

	// 之所以在这两个地方要保存这种相反的旋转，是因为三角化求深度的时候需要这个相反旋转的矩阵
	// 第l帧
	// 将枢纽帧和最后一帧Twc转成Tcw，包括四元数，旋转矩阵，平移向量和增广矩阵
	c_Quat[l] = q[l].inverse(); // 四元数取逆
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	// 滑动窗口中的最新帧
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 

	//以frame_num - 1为参考帧，根据第l和frame_num - 1帧的R,T，三角化一些点，然后再用PNP得到l+1到frame-1之间所有相对pose，然后恢复这些3D点
	
	/** 
	 * 根据第l和frame_num - 1帧的R,T，三角化一些点
	 * 从第l帧到第（frame_num - 2）帧：
	 * 先通过pnp计算第i帧（i = l + 1, ..., frame_num - 2）的位姿（第l帧的位姿已经得到，不再计算）
	 * 再调用triangulateTwoFrames()，与第frame_num - 1帧进行匹配，三角化一些点
	 */
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		// 先调用下面的triangulateTwoFrames()三角化第l帧和当前帧，
		// 再pnp求解，再三角化求解第i帧和当前帧，循环求解l + 1到frame_num - 1所有帧
		if (i > l)
		{
			// 这是依次求解，因此上一帧的位姿是已知量
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];  // 已知第i帧上出现的一些特征点在l系的空间坐标，通过上一帧的旋转平移得到下一帧的旋转平移
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		// 遍历l帧到第（frame_num - 2）帧，寻找与第（frame_num - 1）帧的匹配，三角化特征点
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);  // 第i帧和当前帧进行三角化处理
	}


	//3: triangulate l-----l+1 l+2 ... frame_num -2
	// 遍历（l + 1）帧到第（frame_num - 2）帧，寻找与第l帧的匹配，三角化更多的地图点
	/** 
	 * 从第l + 1帧到第（frame_num - 2）帧：
	 * 第i帧（i = l + 1, ..., frame_num - 2）的位姿前面已经计算过了
	 * 再次调用triangulateTwoFrames()，与第l帧进行匹配，再三角化一些特征点
	 */
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);


	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	/** 
	 * 从第l - 1帧到第0帧：
	 * 先通过pnp计算第i帧（i = l - 1, ..., 0）的位姿
	 * 再调用triangulateTwoFrames()，与第l帧进行匹配，三角化一些特征点
	 */
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}


	//5: triangulate all other points
	/**
	 * 对于sfm_f中没有被三角化的点，进行三角化
	 */
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true) // 该特征点已经三角化，跳过
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	// 对滑动窗口中的所有帧的位姿和3D特征点进行BA优化
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		// 这些都是待优化的参数块
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);  // 在这里，可以发现，仅仅是位姿被优化了，特征点的3D坐标没有被优化
		problem.AddParameterBlock(c_translation[i], 3);
		// l帧是参考系，最新帧的平移也是先验，如果不固定住，原本可观的量会变的不可观
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);  // SetParameterBlockConstant()函数是设置参数块为常数，不被优化
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

	// 只有视觉重投影构成约束，因此遍历所有的特征点，构建约束
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)  // 必须是三角化之后的
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)  // 遍历所有的观测帧，对这些帧建立约束
		{
			int l = sfm_f[i].observation[j].first;
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());
			// 约束了这一帧位姿和3d地图点
    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 // 残差快结构体，核函数，[参数列表]
		}

	}

	/*
	Ceres非线性优化问题求解:
	1.设置求解方式，求解最大时间
	2.如果结果收敛或cost值很小，认为求解成功
	3.将求解结果进行赋值
	*/
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//shur消元有2大作用，一个是在最小二乘中利用H矩阵稀疏的性质进行加速求解，另一个是在sliding window时求解marg掉老帧后的先验信息矩阵
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}

	// 优化结束，把double数组的值返回成对应类型的值
	// 同时Tcw -> Twc
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

