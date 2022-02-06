#include "func.h"

// 相机内参
double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

struct ReprojectionError
{
    ReprojectionError(const cv::Point2f &point_2D)
        : observed_x(point_2D.x), observed_y(point_2D.y) {}

    template <typename T>
    bool operator()(const T *const qvec,
                    const T *const tvec,
                    const T *const point_3D,
                    T *residuals) const
    {
        // 相机系坐标
        T p[3];
        ceres::AngleAxisRotatePoint(qvec, point_3D, p);
        p[0] += tvec[0];
        p[1] += tvec[1];
        p[2] += tvec[2];

        // 归一化平面坐标
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // 像素坐标
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        // 计算重投影误差
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    static ceres::CostFunction *Create(const cv::Point2f &point_2D)
    {
        // 残差维度：2 相机旋转：3 相机位移：3 3D点坐标：3
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
            new ReprojectionError(point_2D)));
    }

    double observed_x;
    double observed_y;
};

// 解析法
class ReprojectionCostFunctor : public ceres::SizedCostFunction<2, 3, 3, 3>
{
public:
    ReprojectionCostFunctor(const cv::Point2f &point_2D)
        : observed_x(point_2D.x), observed_y(point_2D.y) {}

    virtual ~ReprojectionCostFunctor(){};

    // parameters[0]：相机旋转角轴r_cw
    // parameters[1]：相机位移
    // parameters[2]：空间点世界坐标
    virtual bool Evaluate(
        double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 相机系坐标
        double p[3];
        ceres::AngleAxisRotatePoint(parameters[0], parameters[2], p);
        p[0] += parameters[1][0];
        p[1] += parameters[1][1];
        p[2] += parameters[1][2];

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
        double R_cw[9];
        ceres::AngleAxisToRotationMatrix(parameters[0], R_cw);
        if (jacobians != nullptr)
        {
            double z2 = p[2] * p[2];
            if (jacobians[0] != nullptr)
            {
                // 旋转雅克比
                jacobians[0][0] = -fx * p[0] * p[1] / z2;
                jacobians[0][1] = fx + fx * p[0] * p[0] / z2;
                jacobians[0][2] = -fx * p[1] / p[2];
                jacobians[0][3] = -fy - fy * p[1] * p[1] / z2;
                jacobians[0][4] = fy * p[0] * p[1] / z2;
                jacobians[0][5] = fy * p[0] / p[2];
            }
            if (jacobians[1] != nullptr)
            {
                // 位移雅克比
                jacobians[1][0] = fx / p[2];
                jacobians[1][1] = 0;
                jacobians[1][2] = -fx * p[0] / z2;
                jacobians[1][3] = 0;
                jacobians[1][4] = fy / p[2];
                jacobians[1][5] = -fy * p[1] / z2;
                // if (jacobians[2] != nullptr)
                // {
                //     // 空间点雅克比
                //     jacobians[2][0] = jacobians[1][0] * R_cw[0] + jacobians[1][2] * R_cw[6];
                //     jacobians[2][1] = jacobians[1][0] * R_cw[1] + jacobians[1][2] * R_cw[7];
                //     jacobians[2][2] = jacobians[1][0] * R_cw[2] + jacobians[1][2] * R_cw[8];
                //     jacobians[2][3] = jacobians[1][4] * R_cw[3] + jacobians[1][5] * R_cw[6];
                //     jacobians[2][4] = jacobians[1][4] * R_cw[4] + jacobians[1][5] * R_cw[7];
                //     jacobians[2][5] = jacobians[1][4] * R_cw[5] + jacobians[1][5] * R_cw[8];
                //     return true;
                // }
            }
            if (jacobians[2] != nullptr)
            {
                // 空间点雅克比
                double jac10 = fx / p[2];
                double jac12 = -fx * p[0] / z2;
                double jac14 = fy / p[2];
                double jac15 = -fy * p[1] / z2;
                jacobians[2][0] = jac10 * R_cw[0] + jac12 * R_cw[6];
                jacobians[2][1] = jac10 * R_cw[1] + jac12 * R_cw[7];
                jacobians[2][2] = jac10 * R_cw[2] + jac12 * R_cw[8];
                jacobians[2][3] = jac14 * R_cw[3] + jac15 * R_cw[6];
                jacobians[2][4] = jac14 * R_cw[4] + jac15 * R_cw[7];
                jacobians[2][5] = jac14 * R_cw[5] + jac15 * R_cw[8];
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
    std::vector<Eigen::Vector3d> r_result;
    Eigen::AngleAxisd rvec1(cam_poses[0].block<3, 3>(0, 0));
    Eigen::AngleAxisd rvec2(cam_poses[1].block<3, 3>(0, 0));
    r_result.push_back(rvec1.angle() * rvec1.axis());
    r_result.push_back(rvec2.angle() * rvec2.axis());

    std::vector<Eigen::Vector3d> t_result;
    Eigen::Vector3d tvec1 = cam_poses[0].col(3);
    Eigen::Vector3d tvec2 = cam_poses[1].col(3);
    t_result.push_back(tvec1);
    t_result.push_back(tvec2);

    ceres::Problem problem;
    // double cost;
    // std::vector<double> residuals, gradient;
    // ceres::CRSMatrix jacobian;
    // double *cost;
    // double residuals[2];
    // // double jacobian[3][6];
    // double **jacobian = new double *[3];
    // for (int i = 0; i < 3; i++)
    //     jacobian[i] = new double[6];
    for (int i = 0; i < cam_poses.size(); ++i)
    {
        for (int j = 0; j < Points_2D[i].size(); ++j)
        {
            // // 自动求导
            // ceres::CostFunction *cost_function = ReprojectionError::Create(Points_2D[i][j]);

            // 解析法
            ceres::CostFunction *cost_function = new ReprojectionCostFunctor(Points_2D[i][j]);

            // 添加残差块
            auto id = problem.AddResidualBlock(cost_function,
                                               NULL /* squared loss */,
                                               r_result[i].data(), t_result[i].data(), Points_3D[j].data());

            // 检测残差
            // double *parameters[3] = {r_result[i].data(), t_result[i].data(), Points_3D[j].data()};
            // cost_function->Evaluate(parameters, residuals, jacobian);
            // problem.EvaluateResidualBlock(id, false, cost, residuals, jacobian);
            // std::cout << "  " << residuals[0] << "  " << residuals[1] << std::endl;
            // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, &jacobian);
            // std::cout << "problem.Evaluate：" << cost << "  " << residuals[residuals.size() - 2] << "  " << residuals.back() << "  " << std::endl;
        }

        // // 设置参数为四元数
        // ceres::LocalParameterization *quaternion_parameterization =
        //     new ceres::QuaternionParameterization;
        // problem.SetParameterization(q_result[i].data(), quaternion_parameterization);

        // 设置第一帧位姿固定
        if (keep_fixed[i])
        {
            problem.SetParameterBlockConstant(r_result[i].data());
            problem.SetParameterBlockConstant(t_result[i].data());
        }
        else
        {
            // // 设置位移X轴参数固定
            // const std::vector<int> constant_tvec_idxs{0};
            // ceres::SubsetParameterization *tvec_parameterization =
            //     new ceres::SubsetParameterization(3, constant_tvec_idxs);
            // problem.SetParameterization(t_result[i].data(), tvec_parameterization);
        }
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    // 优化
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << t_result[0] << std::endl;
    std::cout << t_result[1] << std::endl;
    return;
}