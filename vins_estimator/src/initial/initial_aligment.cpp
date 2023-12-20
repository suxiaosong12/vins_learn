#include "initial_alignment.h"

//see V-B-1 in Paper
//求解陀螺仪零偏，同时利用求出来的零偏重新进行预积分
//根据视觉SFM的结果来校正陀螺仪的Bias，注意得到了新的Bias后对应的预积分需要repropagate
//求解IMU偏移思路：IMU预计分增量 = SFM相邻位姿变换
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;  // A和b对应的是Ax=b,采用LDLT分解
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;  // frame_i和frame_j分别读取all_image_frame中的相邻两帧
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);  // 构造Ax=b等式
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);  // q_ij = qc0_bk.transpose() * qc0_bk+1
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);  // tmp_A = Jy_bw
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();  // tmp_b = 2 * (ybk_bk+1.inverse() * q_ij)
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;  // 转为正定矩阵，然后用ldlt分解

    }
    delta_bg = A.ldlt().solve(b);  // 采用LDLT分解，求解delta_bg
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++)  // 因为求得的delta_bg只是Bias的变化量，所以需要在滑窗内累加得到Bias的准确值
        Bgs[i] += delta_bg;

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)  // 利用新的Bias重新计算所有帧的IMU积分
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

//Algorithm 1 to find b1 b2
//在半径为G的半球找到切面的一对正交基
//求法跟论文不太一致，但是没影响
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();  //得到一个垂直于a(重力)的矢量b，且模为1
    c = a.cross(b);  // a叉乘b
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

//see V-B-3 in Paper
//1.按照论文思路，重力向量是由重力大小所约束的，论文中使用半球加上半球切面来参数化重力
//2.然后迭代求得w1,w2
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();  // g0 = (g除以原来的模值)*（G的模值9.8）
    Vector3d lx, ly;  //论文中w1,w2
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;  // 对比LinearAlignment()函数中的g，由三自由度变为了两自由度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)  // 迭代4次
    {
        MatrixXd lxly(3, 2);  // lxly = b = [b1,b2]
        lxly = TangentBasis(g0);  // 构建切向空间
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;
            // tmp_A(6,9) = [-I*dt           0             (R^bk_c0)*dt*dt*b/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ] 
            //              [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt*b                  0                    ]
            // tmp_b(6,1) = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c - (R^bk_c0)*dt*dt*||g||*(g^-)/2 , (b^bk_bk+1)-(R^bk_c0)dt*||g||*(g^-)]^T
            // tmp_A * x = tmp_b 求解最小二乘问题

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);  // ldlt分解，得到优化后的状态量x
            VectorXd dg = x.segment<2>(n_state - 3);  // dg = [w1,w2]^T
            //每次normalized（）之后 (g0 + lxly * dg).normalized()  即为每次迭代后得到b1 b2后，只更新重力方向，不更新模值
            // 模值仍然是G.norm()；但是方向更新为g0 + lxly * dg
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

//求解各帧的速度，枢纽帧的重力方向，以及尺度
//初始化滑动窗口中每帧的 速度V[0:n] Gravity Vectorg,尺度s -> 对应论文的V-B-2
//重力修正RefineGravity -> 对应论文的V-B-3
//重力方向跟世界坐标的Z轴对齐
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;  // 需要优化的状态量个数

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;  // frame_i和frame_j分别读入all_frame_count中的相邻两帧
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();  // 没明白干嘛的

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();  // 因为r_A的后3行列为0，所以这么加，又因为前面all_frame_count * 3这是速度的维度
        b.segment<6>(i * 3) += r_b.head<6>();  // b的行数要等于A的，所以这里k=0到n的速度求解时，A和b的矩阵块都要填上

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();  // 求解s的矩阵块
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();  // 求解g的矩阵块
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 乘1000，增强数值稳定性
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);  // ldlt分解把x偏大的100矫正
    double s = x(n_state - 1) / 100.0;  // x(n_state - 1)  指的是x向量的最后一个值的意思
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)  // 如果重力加速度与参考值差距过大或者尺度为负数，则说明计算错误
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x); // 在正切空间微调重力向量
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

// visual-inertial alignment：视觉SFM的结果与IMU预积分结果对齐
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    //估测陀螺仪的Bias，对应论文V-B-1
    solveGyroscopeBias(all_image_frame, Bgs);

    //求解V 重力向量g和 尺度s
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
