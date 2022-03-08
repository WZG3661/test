#include "func.h"

// 相机内参
double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

// 实现四元数参数类
// ceres也有一个QuaternionParameterization，但是计算的是d_q/d_delta_theta，即四元数对扰动量的导数
// 我们已经能够直接计算对扰动量的导数，因此ComputeJacobian()部分可以写成单位阵
class QuaternionParam : public ceres::LocalParameterization
{
public:
    virtual ~QuaternionParam() {}
    bool Plus(const double *x,
              const double *delta,
              double *x_plus_delta) const override;
    bool ComputeJacobian(const double *x, double *jacobian) const override;
    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
};

bool QuaternionParam::Plus(const double *x,
                           const double *delta,
                           double *x_plus_delta) const
{
    const double theta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (theta > 0.0)
    {
        const double sin_theta_by_theta = (sin(0.5 * theta) / theta);
        double q_delta[4];
        q_delta[0] = cos(0.5 * theta);
        q_delta[1] = sin_theta_by_theta * delta[0];
        q_delta[2] = sin_theta_by_theta * delta[1];
        q_delta[3] = sin_theta_by_theta * delta[2];
        ceres::QuaternionProduct(q_delta, x, x_plus_delta); // 左扰动
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            x_plus_delta[i] = x[i];
        }
    }
    return true;
}

bool QuaternionParam::ComputeJacobian(const double *x,
                                      double *jacobian) const
{
    // 已经计算了对delta_theta(扰动量)的雅克比，因此这里写成单位阵即可
    // clang-format off
    jacobian[0] = 0;  jacobian[1]  = 0;   jacobian[2]  = 0;
    jacobian[3] = 1;  jacobian[4]  = 0;   jacobian[5]  = 0;
    jacobian[6] = 0;  jacobian[7]  = 1;   jacobian[8]  = 0;
    jacobian[9] = 0;  jacobian[10] = 0;   jacobian[11] = 1;
    // clang-format on
    return true;
}

// 解析法
class ReprojectionCostFunctor : public ceres::SizedCostFunction<2, 4, 3, 3>
{
public:
    ReprojectionCostFunctor(const cv::Point2f &point_2D)
        : observed_x(point_2D.x), observed_y(point_2D.y) {}

    virtual ~ReprojectionCostFunctor(){};

    // parameters[0]：相机旋转四元数q_cw
    // parameters[1]：相机位移
    // parameters[2]：空间点世界坐标
    virtual bool Evaluate(
        double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 相机系坐标
        double P_C[3], p[3]; // 相机系下空间点坐标、重投影像素坐标
        ceres::QuaternionRotatePoint(parameters[0], parameters[2], P_C);
        p[0] = P_C[0] + parameters[1][0];
        p[1] = P_C[1] + parameters[1][1];
        p[2] = P_C[2] + parameters[1][2];

        // 归一化平面坐标
        double xp = p[0] / p[2];
        double yp = p[1] / p[2];

        // 像素坐标
        double predicted_x = fx * xp + cx;
        double predicted_y = fy * yp + cy;

        // 计算重投影误差
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        // 计算雅克比
        if (jacobians != nullptr)
        {
            // 误差对相机系下投影点的导数
            double z2 = p[2] * p[2];
            Eigen::Matrix<double, 2, 3> Jac_e_Pc;
            Jac_e_Pc << fx / p[2], 0, -fx * p[0] / z2,
                0, fy / p[2], -fy * p[1] / z2;

            if (jacobians[0] != nullptr)
            {
                // 投影点对扰动量的导数
                Eigen::Matrix3d Jac_Pc_dTheta;
                Jac_Pc_dTheta << 0.0, -P_C[2], P_C[1],
                    P_C[2], 0.0, -P_C[0],
                    -P_C[1], P_C[0], 0.0;
                Jac_Pc_dTheta = -Jac_Pc_dTheta;

                // 误差对扰动量的导数
                Eigen::Matrix<double, 2, 3> Jac_e_dTheta = Jac_e_Pc * Jac_Pc_dTheta;

                // 转成2x4，对应四元数
                jacobians[0][0] = 0;
                jacobians[0][1] = Jac_e_dTheta(0, 0);
                jacobians[0][2] = Jac_e_dTheta(0, 1);
                jacobians[0][3] = Jac_e_dTheta(0, 2);
                jacobians[0][4] = 0;
                jacobians[0][5] = Jac_e_dTheta(1, 0);
                jacobians[0][6] = Jac_e_dTheta(1, 1);
                jacobians[0][7] = Jac_e_dTheta(1, 2);
            }
            if (jacobians[1] != nullptr)
            {
                // 位移雅克比
                jacobians[1][0] = Jac_e_Pc(0, 0);
                jacobians[1][1] = Jac_e_Pc(0, 1);
                jacobians[1][2] = Jac_e_Pc(0, 2);
                jacobians[1][3] = Jac_e_Pc(1, 0);
                jacobians[1][4] = Jac_e_Pc(1, 1);
                jacobians[1][5] = Jac_e_Pc(1, 2);
            }
            if (jacobians[2] != nullptr)
            {
                // 空间点雅克比
                Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_cw; // 注意这里RowMajor
                ceres::QuaternionToRotation(parameters[0], R_cw.data());
                Eigen::Matrix<double, 2, 3> Jac_e_P = Jac_e_Pc * R_cw;
                jacobians[2][0] = Jac_e_P(0, 0);
                jacobians[2][1] = Jac_e_P(0, 1);
                jacobians[2][2] = Jac_e_P(0, 2);
                jacobians[2][3] = Jac_e_P(1, 0);
                jacobians[2][4] = Jac_e_P(1, 1);
                jacobians[2][5] = Jac_e_P(1, 2);
            }
        }
        return true;
    }

    double observed_x;
    double observed_y;
};

void poseOptimisition(std::vector<Eigen::Matrix<double, 3, 4>> &cam_poses,
                      std::vector<std::vector<cv::Point2f>> &Points_2D,
                      std::vector<Eigen::Vector3d> &Points_3D,
                      std::vector<bool> &keep_fixed)
{
    google::InitGoogleLogging("poseOptimisition");
    std::vector<Eigen::Vector4d> q_result;
    Eigen::Quaterniond q1(cam_poses[0].block<3, 3>(0, 0));
    Eigen::Quaterniond q2(cam_poses[1].block<3, 3>(0, 0));
    Eigen::Vector4d qvec1(q1.w(), q1.x(), q1.y(), q1.z());
    Eigen::Vector4d qvec2(q2.w(), q2.x(), q2.y(), q2.z());
    q_result.push_back(qvec1);
    q_result.push_back(qvec2);

    std::vector<Eigen::Vector3d> t_result;
    Eigen::Vector3d tvec1 = cam_poses[0].col(3);
    Eigen::Vector3d tvec2 = cam_poses[1].col(3);
    t_result.push_back(tvec1);
    t_result.push_back(tvec2);

    double cost;
    double residuals[2];
    double jac1[20] = {0}, jac2[20] = {0}, jac3[20] = {0};
    double *jacobians[3] = {jac1, jac2, jac3};
    ceres::Problem problem;
    for (int i = 0; i < cam_poses.size(); ++i)
    {
        for (int j = 0; j < Points_2D[i].size(); ++j)
        {
            // 解析法
            ceres::CostFunction *cost_function = new ReprojectionCostFunctor(Points_2D[i][j]);

            // Loss
            ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

            // 添加残差块
            auto id = problem.AddResidualBlock(cost_function, loss_function,
                                               q_result[i].data(), t_result[i].data(), Points_3D[j].data());

            // // 检测残差
            // problem.EvaluateResidualBlock(id, true, &cost, residuals, jacobians);
            // std::cout << cost << "  " << residuals[0] << "  " << residuals[1] << std::endl;
        }

        // 设置参数为四元数
        ceres::LocalParameterization *quaternion_parameterization = new QuaternionParam;
        problem.SetParameterization(q_result[i].data(), quaternion_parameterization);

        // 设置第一帧位姿固定
        if (keep_fixed[i])
        {
            problem.SetParameterBlockConstant(q_result[i].data());
            problem.SetParameterBlockConstant(t_result[i].data());
        }
    }

    // 配置求解器
    ceres::Solver::Options options;
    // options.check_gradients = true;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;

    // 优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "优化后结果为：" << std::endl;
    std::cout << q_result[0] << std::endl;
    std::cout << q_result[1] << std::endl;
    std::cout << t_result[0] << std::endl;
    std::cout << t_result[1] << std::endl;

    return;
}