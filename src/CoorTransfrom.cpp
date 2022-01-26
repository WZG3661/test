#include "func.h"

void RCam2LCam()
{
    // 左相机到IMU
    Eigen::Vector3d p_Imu_LCam(-0.050720060477640147, -0.0017414170413474165, 0.0022943667597148118);
    Eigen::Quaterniond q_Imu_LCam(0.99090224973327068, 0.13431639597354814,
                                  0.00095051670014565813, -0.0084222184858180373);
    q_Imu_LCam.normalize();

    // 右相机到IMU
    Eigen::Vector3d p_Imu_RCam(0.051932496584961352, -0.0011555929083120534, 0.0030949732069645722);
    Eigen::Quaterniond q_Imu_RCam(0.99073762672679389, 0.13492462817073628,
                                  -0.00013648999867379373, -0.015306242884176362);
    q_Imu_RCam.normalize();

    // 计算右相机到左相机的外参
    Eigen::Quaterniond q_LCam_RCam = q_Imu_LCam * q_Imu_RCam.inverse();
    Eigen::Vector3d p_LCam_RCam = q_Imu_LCam * (p_Imu_RCam - p_Imu_LCam);

    std::cout << q_LCam_RCam << std::endl;
    std::cout << p_LCam_RCam << std::endl;
}