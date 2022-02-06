#include "func.h"

using namespace cv;

void imgInverse(cv::Mat img);

void imgUndistort(const cv::Mat &input, cv::Mat &output, const cv::Mat &K, const cv::Mat &D);

void imgMatching(const cv::Mat &img1, const cv::Mat &img2,
                 std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2, const cv::Mat &K);

void calcPose(const cv::Mat &K,
              std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
              Eigen::Matrix<double, 3, 4> &T1, Eigen::Matrix<double, 3, 4> &T2,
              std::vector<Eigen::Vector3d> &Points_3D);

void triangulatePoints(const Eigen::Matrix<double, 3, 4> &P0, const Eigen::Matrix<double, 3, 4> &P1,
                       const std::vector<cv::Point2f> &pt0, const std::vector<cv::Point2f> &pt1,
                       std::vector<Eigen::Vector3d> &Points);

void wld2cam(const Eigen::Matrix<double, 3, 4> &T_CW,
             const std::vector<Eigen::Vector3d> &Points_W, std::vector<Eigen::Vector3d> &Points_C);

void imgProcess(const std::string &path1, const std::string &path2,
                Eigen::Matrix<double, 3, 4> &T1, Eigen::Matrix<double, 3, 4> &T2,
                std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
                std::vector<Eigen::Vector3d> &Points_3D)
{
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);

    // 内参
    const cv::Mat K = (cv::Mat_<double>(3, 3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
    // 畸变系数
    const cv::Mat D = (cv::Mat_<double>(4, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    // 图像像素取反
    // imgInverse(img1);

    // 图像去畸变
    cv::Mat new_img1, new_img2;
    imgUndistort(img1, new_img1, K, D);
    imgUndistort(img2, new_img2, K, D);

    // 特征匹配
    // std::vector<cv::Point2f> p1, p2;
    imgMatching(new_img1, new_img2, p1, p2, K);

    // 恢复位姿
    calcPose(K, p1, p2, T1, T2, Points_3D);
}

// 对图像每个像素取反
void imgInverse(cv::Mat img)
{
    cv::Mat img_copy = img.clone();
    int width = img_copy.cols;
    int height = img_copy.rows;

    for (size_t i = 0; i < height; ++i)
    {
        u_char *row_ptr = img_copy.ptr<u_char>(i);
        for (size_t j = 0; j < width; ++j)
        {
            row_ptr[j] = 255 - row_ptr[j];
        }
    }
    cv::Mat img_merge;
    img_merge.push_back(img);
    img_merge.push_back(img_copy);
    cv::imshow("img", img_merge);
    cv::waitKey(0);
}

// 对图像去畸变
void imgUndistort(const cv::Mat &input, cv::Mat &output, const cv::Mat &K, const cv::Mat &D)
{

    // 使用undistort
    cv::Mat UndistortImage1;
    cv::undistort(input, UndistortImage1, K, D, K);
    // cv::imshow("undistort", UndistortImage1);
    // cv::imwrite("/home/nreal/Test/data/two_image_pose_estimation/undistort.png", UndistortImage1);

    // 使用remap
    cv::Size imgSize(input.cols, input.rows);
    double alpha = 0;
    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, imgSize, alpha, imgSize, 0);
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(K, D, Mat_<double>::eye(3, 3), NewCameraMatrix, imgSize, CV_16SC2, map1, map2);
    // map1和map2表示图像对的像素映射关系，大小与原图相同
    cv::Mat UndistortImage2;
    remap(input, UndistortImage2, map1, map2, cv::INTER_LINEAR);
    // cv::imshow("remap", UndistortImage2);
    // cv::imwrite("/home/nreal/Test/data/two_image_pose_estimation/remap.png", UndistortImage2);

    // cv::Mat img_merge;
    // cv::hconcat(img, UndistortImage1, img_merge);
    // cv::hconcat(img_merge, UndistortImage2, img_merge);
    // cv::imshow("undistorted", img_merge);
    // cv::waitKey(0);

    output = UndistortImage1.clone();
    return;
}

void imgMatching(const cv::Mat &img1, const cv::Mat &img2,
                 std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2, const cv::Mat &K)
{
    // 初始化
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::FeatureDetector> detector = ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = ORB::create();
    cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 角点检测
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // 计算描述子
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    // 匹配
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    int matches_num = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        matches_num++;

        p1.push_back(keypoints1[matches[i].queryIdx].pt);
        p2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    std::cout << p1.size() << std::endl;

    // 使用RANSAC筛选内点
    std::vector<u_char> mask;
    // Mat Fundamental = cv::findFundamentalMat(p1, p2, FM_RANSAC, 3, 0.99, mask);
    cv::Mat E = cv::findEssentialMat(p1, p2, K, RANSAC, 0.9999999, 1, mask);
    // std::cout << E << std::endl;
    matches_num = 0;
    p1.clear();
    p2.clear();
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (mask[i] != 0)
        {
            p1.push_back(keypoints1[matches[i].queryIdx].pt);
            p2.push_back(keypoints2[matches[i].trainIdx].pt);
            matches_num++;
        }
    }

    // OpenCV计算位姿
    cv::Mat R, t;
    cv::recoverPose(E, p1, p2, K, R, t);
    std::cout << R << std::endl;
    std::cout << t << std::endl;
    cv::Quatd q = cv::Quatd::createFromRotMat(R);
    std::cout << q << std::endl;

    // // 显示匹配效果
    // cv::Mat img_merge;
    // cv::hconcat(img1, img2, img_merge);
    // cv::cvtColor(img_merge, img_merge, COLOR_GRAY2BGR);
    // for (int i = 0; i < matches_num; ++i)
    // {
    //     cv::Point2f start(p1[i].x + img1.cols, p1[i].y);
    //     cv::Point2f end(p2[i].x + img1.cols, p2[i].y);
    //     cv::circle(img_merge, p1[i], 3, cv::Scalar(0, 255, 0));
    //     cv::circle(img_merge, end, 3, cv::Scalar(0, 255, 0));
    //     cv::line(img_merge, p1[i], p2[i], cv::Scalar(0, 255, 255));
    //     cv::line(img_merge, start, end, cv::Scalar(0, 255, 255));
    // }
    // cv::imshow("Matches", img_merge);
    // // cv::imwrite("/home/nreal/Test/data/two_image_pose_estimation/Matches_by_RANSAC.png", img_merge);
    // cv::waitKey(0);
}

void calcPose(const cv::Mat &K,
              std::vector<cv::Point2f> &p1, std::vector<cv::Point2f> &p2,
              Eigen::Matrix<double, 3, 4> &T1_ret, Eigen::Matrix<double, 3, 4> &T2_ret,
              std::vector<Eigen::Vector3d> &Points_3D)
{
    int N = p1.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
    A.resize(N, 9);

    // 相机内参
    float fx = K.ptr<double>(0)[0];
    float fy = K.ptr<double>(1)[1];
    float cx = K.ptr<double>(0)[2];
    float cy = K.ptr<double>(1)[2];

    // 计算E
    std::vector<cv::Point2f> pt1, pt2;
    for (int i = 0; i < N; ++i)
    {
        const float u1 = p1[i].x;
        const float v1 = p1[i].y;
        const float u2 = p2[i].x;
        const float v2 = p2[i].y;

        // 计算归一化坐标
        const float x1 = (u1 - cx) / fx;
        const float y1 = (v1 - cy) / fy;
        const float x2 = (u2 - cx) / fx;
        const float y2 = (v2 - cy) / fy;

        pt1.push_back(cv::Point2f(x1, y1));
        pt2.push_back(cv::Point2f(x2, y2));

        A(i, 0) = x2 * x1;
        A(i, 1) = x2 * y1;
        A(i, 2) = x2;
        A(i, 3) = y2 * x1;
        A(i, 4) = y2 * y1;
        A(i, 5) = y2;
        A(i, 6) = x1;
        A(i, 7) = y1;
        A(i, 8) = 1;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto V = svd_A.matrixV();
    Eigen::MatrixXd E = V.col(V.cols() - 1);
    E.resize(3, 3);
    E.transposeInPlace();
    // std::cout << E << std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto singVal = svd_E.singularValues();
    double sigma = (singVal(0) + singVal(1)) * 0.5;
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    W.diagonal() << sigma, sigma, 0;
    E = svd_E.matrixU() * W * svd_E.matrixV().transpose();
    // std::cout << E << std::endl;

    // 分解E
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_E2(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    W = Eigen::Matrix3d::Zero();
    W.diagonal() = svd_E2.singularValues();
    Eigen::Matrix3d Rz;
    Rz << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;
    Eigen::Matrix3d R1 = svd_E2.matrixU() * Rz * svd_E2.matrixV().transpose();
    Eigen::Matrix3d R2 = svd_E2.matrixU() * Rz.transpose() * svd_E2.matrixV().transpose();
    Eigen::Matrix3d t_mat = svd_E2.matrixU() * Rz * W * svd_E2.matrixU().transpose();
    Eigen::Vector3d t1, t2;
    t1 << t_mat(2, 1), t_mat(0, 2), t_mat(1, 0);
    t1.normalize();
    t2 = -t1;
    // std::cout << R1 << std::endl;
    // std::cout << R2 << std::endl;

    // 恢复位姿
    Eigen::Matrix<double, 3, 4> T0;
    T0 << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();

    Eigen::Matrix<double, 3, 4> T1, T2, T3, T4;
    T1 << R1, t1;
    T2 << R2, t1;
    T3 << R1, t2;
    T4 << R2, t2;
    std::vector<Eigen::Vector3d> Points1_0, Points2_0, Points3_0, Points4_0;
    std::vector<Eigen::Vector3d> Points1_1, Points2_1, Points3_1, Points4_1;
    triangulatePoints(T0, T1, pt1, pt2, Points1_0);
    triangulatePoints(T0, T2, pt1, pt2, Points2_0);
    triangulatePoints(T0, T3, pt1, pt2, Points3_0);
    triangulatePoints(T0, T4, pt1, pt2, Points4_0);
    wld2cam(T1, Points1_0, Points1_1);
    wld2cam(T2, Points2_0, Points2_1);
    wld2cam(T3, Points3_0, Points3_1);
    wld2cam(T4, Points4_0, Points4_1);
    auto positive_depth = [](Eigen::Vector3d pt)
    { return pt(2) > 0; };
    int num1 = std::count_if(Points1_0.begin(), Points1_0.end(), positive_depth);
    int num2 = std::count_if(Points2_0.begin(), Points2_0.end(), positive_depth);
    int num3 = std::count_if(Points3_0.begin(), Points3_0.end(), positive_depth);
    int num4 = std::count_if(Points4_0.begin(), Points4_0.end(), positive_depth);
    num1 += std::count_if(Points1_1.begin(), Points1_1.end(), positive_depth);
    num2 += std::count_if(Points2_1.begin(), Points2_1.end(), positive_depth);
    num3 += std::count_if(Points3_1.begin(), Points3_1.end(), positive_depth);
    num4 += std::count_if(Points4_1.begin(), Points4_1.end(), positive_depth);
    int max = std::max(std::max(std::max(num1, num2), num3), num4);
    if (max == num1)
    {
        T2_ret << R1, t1;
        Points_3D = Points1_0;
    }
    else if (max == num2)
    {
        T2_ret << R2, t1;
        Points_3D = Points2_0;
    }
    else if (max == num3)
    {
        T2_ret << R1, t2;
        Points_3D = Points3_0;
    }
    else
    {
        T2_ret << R2, t2;
        Points_3D = Points4_0;
    }
    T1_ret = T0;

    std::cout << "位姿：\n"
              << T2_ret << std::endl;
}

void triangulatePoints(const Eigen::Matrix<double, 3, 4> &T0, const Eigen::Matrix<double, 3, 4> &T1,
                       const std::vector<cv::Point2f> &pt0, const std::vector<cv::Point2f> &pt1, std::vector<Eigen::Vector3d> &Points)
{
    for (int i = 0; i < pt0.size(); ++i)
    {
        Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
        A.row(0) = pt0[i].y * T0.row(2) - T0.row(1);
        A.row(1) = -pt0[i].x * T0.row(2) + T0.row(0);
        A.row(2) = pt1[i].y * T1.row(2) - T1.row(1);
        A.row(3) = -pt1[i].x * T1.row(2) + T1.row(0);
        Eigen::Vector4d point = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
        // if (std::abs(point(3)) < FLT_EPSILON)
        //     point(3) *= 10;
        Eigen::Vector3d pt(point(0) / point(3), point(1) / point(3), point(2) / point(3));
        // std::cout << pt << std::endl;
        Points.push_back(pt);
    }
}

void wld2cam(const Eigen::Matrix<double, 3, 4> &T_CW,
             const std::vector<Eigen::Vector3d> &Points_W, std::vector<Eigen::Vector3d> &Points_C)
{
    for (int i = 0; i < Points_W.size(); ++i)
    {
        Eigen::Vector4d point_W;
        point_W << Points_W[i][0], Points_W[i][1], Points_W[i][2], 1.0;
        // Points_C.push_back(T_CW * point_W);
        Points_C.push_back(T_CW * point_W);
    }
}