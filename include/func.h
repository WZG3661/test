#ifndef _FUNC_H_
#define _FUNC_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
// #include <opencv2/core/core.hpp>
#include <opencv2/core/quaternion.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

void func();
void RCam2LCam();
void curveFit();
void imgProcess(const std::string &path1, const std::string &path2,
                Eigen::Matrix<double, 3, 4> &T1, Eigen::Matrix<double, 3, 4> &T2,
                std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
                std::vector<Eigen::Vector3d> &Points_3D);
void poseOptimisition(std::vector<Eigen::Matrix<double, 3, 4>> &cam_poses,
                      std::vector<std::vector<cv::Point2f>> &Points_2D,
                      std::vector<Eigen::Vector3d> &Points_3D,
                      std::vector<bool> &keep_fixed);
#endif