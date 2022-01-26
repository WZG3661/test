#include "func.h"
using namespace std;
using namespace Eigen;

void curveFit1();
void curveFit2();

std::vector<double> x, y_noise;

void readData(std::string &path)
{
    std::ifstream srcFile(path);
    if (!srcFile)
    {
        std::cout << "error opening source file." << std::endl;
        return;
    }

    x.clear();
    y_noise.clear();

    // 读文件
    char line[50];
    srcFile.getline(line, 50);
    std::stringstream ss;
    while (srcFile.getline(line, 50))
    {
        ss << line;
        double x_temp, y_temp;
        ss >> x_temp >> y_temp;
        ss.clear();

        x.push_back(x_temp);
        y_noise.push_back(y_temp);
    }
}

void curveFit()
{
    std::string path1 = "/home/nreal/Test/data.txt";
    readData(path1);
    curveFit1();

    std::string path2 = "/home/nreal/Test/data2.txt";
    readData(path2);
    curveFit2();
}

void curveFit1()
{
    // 形成H和b
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    H.resize(x.size(), 2);
    b.resize(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        H(i, 0) = x[i];
        H(i, 1) = 1;
        b(i) = y_noise[i];
    }

    // 普通方法
    std::cout << "The solution using normal equations is:\n"
              << (H.transpose() * H).ldlt().solve(H.transpose() * b) << std::endl;

    // QR 分解
    std::cout << "The solution using the QR decomposition is:\n"
              << H.colPivHouseholderQr().solve(b) << std::endl;

    // SVD 分解
    std::cout << "The least-squares solution is:\n"
              << H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
}

void curveFit2()
{
    // 形成H和b
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    H.resize(x.size(), 2);
    b.resize(x.size());
    for (int i = 0; i < x.size(); ++i)
    {
        H(i, 0) = x[i];
        H(i, 1) = 1;
        b(i) = y_noise[i];
    }

    // 普通方法
    std::cout << "The solution using normal equations is:\n"
              << (H.transpose() * H).ldlt().solve(H.transpose() * b) << std::endl;

    // QR 分解
    std::cout << "The solution using the QR decomposition is:\n"
              << H.colPivHouseholderQr().solve(b) << std::endl;

    // SVD 分解
    std::cout << "The least-squares solution is:\n"
              << H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
}