#include <iostream>
#include "func.h"
#include "libtest.h"

int main()
{
    // test lib
    libFunc(3);
    func();

    // CoorTransform
    // RCam2LCam();

    // CurveFit
    // curveFit();

    // image process
    std::string path1 = "/home/nreal/Test/data/two_image_pose_estimation/1403637188088318976.png";
    std::string path2 = "/home/nreal/Test/data/two_image_pose_estimation/1403637189138319104.png";
    Eigen::Matrix<double, 3, 4> T1, T2;
    std::vector<cv::Point2f> p1, p2;
    std::vector<Eigen::Vector3d> Points_3D;
    imgProcess(path1, path2, T1, T2, p1, p2, Points_3D);

    // ceres
    std::vector<Eigen::Matrix<double, 3, 4>> cam_poses{T1, T2};
    std::vector<std::vector<cv::Point2f>> Points_2D{p1, p2};
    std::vector<bool> keep_fixed = {true, false};
    poseOptimisition(cam_poses, Points_2D, Points_3D, keep_fixed);
    // std::cout << "vdsg" << std::endl;
    return 0;
}