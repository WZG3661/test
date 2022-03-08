// #include "func.h"

// // 相机内参
// double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

// struct ReprojectionError
// {
//     ReprojectionError(const cv::Point2f &point_2D)
//         : observed_x(point_2D.x), observed_y(point_2D.y) {}

//     template <typename T>
//     bool operator()(const T *const qvec,
//                     const T *const tvec,
//                     const T *const point_3D,
//                     T *residuals) const
//     {
//         // 相机系坐标
//         T p[3];
//         ceres::QuaternionRotatePoint(qvec, point_3D, p);
//         p[0] += tvec[0];
//         p[1] += tvec[1];
//         p[2] += tvec[2];

//         // 归一化平面坐标
//         T xp = p[0] / p[2];
//         T yp = p[1] / p[2];

//         // 像素坐标
//         T predicted_x = T(fx) * xp + T(cx);
//         T predicted_y = T(fy) * yp + T(cy);

//         // 计算重投影误差
//         residuals[0] = predicted_x - observed_x;
//         residuals[1] = predicted_y - observed_y;

//         return true;
//     }

//     static ceres::CostFunction *Create(const cv::Point2f &point_2D)
//     {
//         // 残差维度：2 相机旋转：4 相机位移：3 3D点坐标：3
//         return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
//             new ReprojectionError(point_2D)));
//     }

//     double observed_x;
//     double observed_y;
// };

// void poseOptimisition(std::vector<Eigen::Matrix<double, 3, 4>> &cam_poses,
//                       std::vector<std::vector<cv::Point2f>> &Points_2D,
//                       std::vector<Eigen::Vector3d> &Points_3D,
//                       std::vector<bool> &keep_fixed)
// {
//     google::InitGoogleLogging("poseOptimisition");
//     std::vector<Eigen::Vector4d> q_result;
//     Eigen::Quaterniond q1(cam_poses[0].block<3, 3>(0, 0));
//     Eigen::Quaterniond q2(cam_poses[1].block<3, 3>(0, 0));
//     Eigen::Vector4d qvec1(q1.w(), q1.x(), q1.y(), q1.z());
//     Eigen::Vector4d qvec2(q2.w(), q2.x(), q2.y(), q2.z());
//     q_result.push_back(qvec1);
//     q_result.push_back(qvec2);

//     std::vector<Eigen::Vector3d> t_result;
//     Eigen::Vector3d tvec1 = cam_poses[0].col(3);
//     Eigen::Vector3d tvec2 = cam_poses[1].col(3);
//     t_result.push_back(tvec1);
//     t_result.push_back(tvec2);

//     ceres::Problem problem;
//     double cost;
//     double residuals[2];
//     double jac1[6], jac2[6], jac3[6];
//     double *jacobians[3] = {jac1, jac2, jac3};

//     for (int i = 0; i < cam_poses.size(); ++i)
//     {
//         for (int j = 0; j < Points_2D[i].size(); ++j)
//         {
//             // 自动求导
//             ceres::CostFunction *cost_function = ReprojectionError::Create(Points_2D[i][j]);

//             // Loss
//             ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

//             // 添加残差块
//             auto id = problem.AddResidualBlock(cost_function, loss_function,
//                                                q_result[i].data(), t_result[i].data(), Points_3D[j].data());

//             // 检测残差
//             problem.EvaluateResidualBlock(id, true, &cost, residuals, jacobians);
//             std::cout << cost << "  " << residuals[0] << "  " << residuals[1] << std::endl;
//         }

//         // 设置参数为四元数
//         ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
//         problem.SetParameterization(q_result[i].data(), quaternion_parameterization);

//         // 设置第一帧位姿固定
//         if (keep_fixed[i])
//         {
//             problem.SetParameterBlockConstant(q_result[i].data());
//             problem.SetParameterBlockConstant(t_result[i].data());
//         }
//         else
//         {

//             // // 设置位移X轴参数固定
//             // const std::vector<int> constant_tvec_idxs{0};
//             // ceres::SubsetParameterization *tvec_parameterization =
//             //     new ceres::SubsetParameterization(3, constant_tvec_idxs);
//             // problem.SetParameterization(t_result[i].data(), tvec_parameterization);
//         }
//     }

//     // 配置求解器
//     ceres::Solver::Options options;
//     options.minimizer_progress_to_stdout = true;
//     options.linear_solver_type = ceres::DENSE_SCHUR;

//     // 优化
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     std::cout << summary.FullReport() << "\n";
//     std::cout << q_result[0] << std::endl;
//     std::cout << q_result[1] << std::endl;
//     std::cout << t_result[0] << std::endl;
//     std::cout << t_result[1] << std::endl;

//     return;
// }
