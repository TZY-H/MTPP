#ifndef __GEN3_H
#define __GEN3_H
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <cmath>

// using namespace std;
using namespace Eigen;
long long utime_ns(void);

class uGen3Ctrl
{
private:
    /* data */
    double QM_d1 = 0.02;
    double QM_d2 = 0.04;
    MatrixXd Gen3_7_T_C(void);
    MatrixXd Gen3_7_T_D(void);

public:
    MatrixXd Gen3_B_T_1(double q);
    MatrixXd Gen3_1_T_2(double q);
    MatrixXd Gen3_2_T_3(double q);
    MatrixXd Gen3_3_T_4(double q);
    MatrixXd Gen3_4_T_5(double q);
    MatrixXd Gen3_5_T_6(double q);
    MatrixXd Gen3_6_T_7(double q);
    MatrixXd Gen3_7_T_H(void);
    MatrixXd Gen3_H_T_D(void);
    
    VectorXd Gen3_QVec;
    std::vector<Matrix4d> Gen3_TMat;
    std::vector<Matrix4d> Gen3_TMatSen;
    MatrixXd Gen3_JacobiMat;
    MatrixXd Gen3_UpBtoH(Matrix<double, 7, 1> Q, uint8_t upkey = 0);
    MatrixXd Gen3_UpBtoHSen(Matrix<double, 7, 1> Q, uint8_t upkey = 0);
    MatrixXd Gen3_Jacobi(VectorXd Q, uint8_t upkey = 0);
    double Gen3_Manipulability(VectorXd Q);
    MatrixXd Gen3_MangGradient(VectorXd Q, double manipu = -1);
    MatrixXd Gen3_Null(MatrixXd A);
    VectorXd Gen3_Qgo(VectorXd Q, VectorXd v);
    VectorXd Gen3_QMgo(VectorXd Q, VectorXd v, double Ratio = 0);
    double Gen3_ArmAngle(VectorXd Q);
    VectorXd Gen3_ArmAngleGradient(VectorXd Q, double armangle = 100);
    VectorXd Gen3_QAgo(VectorXd Q, VectorXd v, double minAngle = 0.34888, double maxAngle = 1.39555, double Ratio = 0);

    uGen3Ctrl(/* args */);
    ~uGen3Ctrl();
};



#endif

