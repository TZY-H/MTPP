#include "uGen3Ctrl.h"

using namespace std;
using namespace Eigen;
long long utime_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<long long>(1000000000UL) * static_cast<long long>(ts.tv_sec) +
           static_cast<long long>(ts.tv_nsec);
}

uGen3Ctrl::uGen3Ctrl(/* args */)
{
    Gen3_TMat.resize(9);
    Gen3_TMatSen.resize(2);
}

uGen3Ctrl::~uGen3Ctrl()
{
}

MatrixXd uGen3Ctrl::Gen3_B_T_1(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        -sin(q), -cos(q), 0, 0,
        0, 0, -1, 0.1564,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_1_T_2(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, -1, 0.0054,
        sin(q), cos(q), 0, -0.1284,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_2_T_3(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, 1, -0.2104,
        -sin(q), -cos(q), 0, -0.0064,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_3_T_4(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, -1, 0.0064,
        sin(q), cos(q), 0, -0.2104,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_4_T_5(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, 1, -0.2084,
        -sin(q), -cos(q), 0, -0.0064,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_5_T_6(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, -1, 0,
        sin(q), cos(q), 0, -0.1059,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_6_T_7(double q)
{
    MatrixXd T(4, 4);
    T << cos(q), -sin(q), 0, 0,
        0, 0, 1, -0.1059,
        -sin(q), -cos(q), 0, 0,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_7_T_H(void)
{
    MatrixXd T(4, 4);
    T << 1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, -1, -0.0615,
        0, 0, 0, 1;
    return T;
}

// 机械夹抓
MatrixXd uGen3Ctrl::Gen3_H_T_D(void)
{
    MatrixXd T(4, 4);
    T << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0.06,
        0, 0, 0, 1;
    return T;
}

// // 发光点
// MatrixXd uGen3Ctrl::Gen3_H_T_D(void)
// {
//     MatrixXd T(4, 4);
//     T << 1, 0, 0, -0.0060,
//         0, 1, 0, 0.1100,
//         0, 0, 1, 0.0135,
//         0, 0, 0, 1;
//     return T;
// }

// 标定的筷子末端
// MatrixXd uGen3Ctrl::Gen3_H_T_D(void)
// {
//     MatrixXd T(4, 4);
//     T << 1, 0, 0, -0.00191553,
//         0, 1, 0, 0.00337776,
//         0, 0, 1, 0.24024879,
//         0, 0, 0, 1;
//     return T;
// }

// 摄像头
MatrixXd uGen3Ctrl::Gen3_7_T_C(void)
{
    MatrixXd T(4, 4);
    T << 1, 0, 0, 0,
        0, -1, 0, -0.05639,
        0, 0, -1, -0.05845,
        0, 0, 0, 1;
    return T;
}
MatrixXd uGen3Ctrl::Gen3_7_T_D(void)
{
    MatrixXd T(4, 4);
    T << 1, 0, 0, 0.0275,
        0, -1, 0, -0.066,
        0, 0, -1, -0.05845,
        0, 0, 0, 1;
    return T;
}

MatrixXd uGen3Ctrl::Gen3_UpBtoH(Matrix<double, 7, 1> Q, uint8_t upkey)
{
    Gen3_QVec = Q;
    MatrixXd outmat = Gen3_B_T_1(Q[0]);
    vector<MatrixXd> mat_1_H = {Gen3_1_T_2(Q[1]), Gen3_2_T_3(Q[2]), Gen3_3_T_4(Q[3]), Gen3_4_T_5(Q[4]),
                                Gen3_5_T_6(Q[5]), Gen3_6_T_7(Q[6]), Gen3_7_T_H(), Gen3_H_T_D()};
    // for (auto tempmat : mat_1_H)
    if (upkey)
    {
        for (int i = 0; i < 8; i++)
        {
            Gen3_TMat[i] = outmat;
            outmat = outmat * mat_1_H[i];
        }
        Gen3_TMat[8] = outmat;
    }
    else
    {
        for (int i = 0; i < 8; i++)
            outmat = outmat * mat_1_H[i];
    }
    return outmat;
}

MatrixXd uGen3Ctrl::Gen3_UpBtoHSen(Matrix<double, 7, 1> Q, uint8_t upkey)
{
    Gen3_UpBtoH(Q, 1);
    MatrixXd outmat = Gen3_TMat[6];
    MatrixXd Cmat = outmat * Gen3_7_T_C();
    MatrixXd Dmat = outmat * Gen3_7_T_D();
    if (upkey)
    {
        Gen3_TMatSen[0] = Cmat;
        Gen3_TMatSen[1] = Dmat;
    }
    return Cmat;
}

MatrixXd uGen3Ctrl::Gen3_Jacobi(VectorXd Q, uint8_t upkey)
{
    double dangle = 1e-6;
    MatrixXd nowmat = Gen3_UpBtoH(Q);
    Vector3d p = nowmat.block<3, 1>(0, 3);
    Matrix3d R = nowmat.block<3, 3>(0, 0);
    Matrix3d R_inv = R.transpose();
    MatrixXd dmat(6, 7);
    dmat = MatrixXd::Zero(6, 7);
    for (int i = 0; i < Q.size(); i++)
    {
        VectorXd dQ = Q;
        dQ[i] = dQ[i] + dangle;
        MatrixXd mat_ = Gen3_UpBtoH(dQ);
        Vector3d dp = (mat_.block<3, 1>(0, 3) - p) / dangle;
        Matrix3d R_ = mat_.block<3, 3>(0, 0);
        Matrix3d dR = R_ * R_inv;
        double angle = std::acos((dR.trace() / 2) - 0.5);
        Matrix3d rmat = (dR - dR.transpose()) / 2 / sin(angle);
        Vector3d vectorR(angle / dangle * rmat(2, 1), angle / dangle * rmat(0, 2), angle / dangle * rmat(1, 0));
        dmat.col(i) << dp, vectorR;
    }
    if (upkey)
        Gen3_JacobiMat = dmat;
    return dmat;
}

double uGen3Ctrl::Gen3_Manipulability(VectorXd Q)
{
    MatrixXd jacobi(6, 7);
    jacobi = Gen3_Jacobi(Q);
    // double detj2 = jacobi.determinant() * jacobi.transpose().determinant();
    // MatrixXd jj = jacobi * jacobi.transpose();
    // double detj2 = jj.determinant();
    double detj2 = (jacobi * jacobi.transpose()).determinant();
    if (abs(detj2) > 1e-41)
    {
        return sqrt(detj2);
    }
    else
    {
        return 0;
    }
}

MatrixXd uGen3Ctrl::Gen3_MangGradient(VectorXd Q, double manipu)
{
    double dangle = 1e-6;
    if (manipu < 0)
        manipu = Gen3_Manipulability(Q);
    MatrixXd dmanipu(1, 7);
    for (int i = 0; i < Q.size(); i++)
    {
        VectorXd dQ = Q;
        dQ[i] = dQ[i] + dangle;
        double manipu_ = Gen3_Manipulability(dQ);
        dmanipu(0, i) = (manipu_ - manipu) / dangle;
    }
    return dmanipu;
}

MatrixXd uGen3Ctrl::Gen3_Null(MatrixXd A)
{
    JacobiSVD<MatrixXd> svd(A, ComputeFullV); // 对输入矩阵A进行奇异值分解。参数ComputeFullV表示要计算矩阵A的全部右奇异向量（V矩阵）。
    VectorXd singular_values = svd.singularValues();
    int rank = 0;
    for (int i = 0; i < singular_values.size(); i++)
    {
        if (singular_values(i) > 1e-15)
        {
            rank++;
        }
    }
    MatrixXd null_space(A.cols(), A.cols() - rank); // 储零空间的基向量
    for (int i = 0; i < A.cols() - rank; i++)
    {
        null_space.col(i) = svd.matrixV().col(rank + i).normalized();
    }
    return null_space;
}

VectorXd uGen3Ctrl::Gen3_Qgo(VectorXd Q, VectorXd v)
{
    MatrixXd jacobi = Gen3_Jacobi(Q);
    MatrixXd jacobi_null = Gen3_Null(jacobi).transpose();
    MatrixXd jacobi_new(jacobi.rows() + jacobi_null.rows(), jacobi.cols());
    jacobi_new << jacobi, jacobi_null;

    VectorXd v_new(v.size() + jacobi_null.rows());
    for (int i = 0; i < v.size(); i++)
    {
        v_new(i) = v[i];
    }
    for (int i = 0; i < jacobi_null.rows(); i++)
    {
        v_new(v.size() + i) = 0;
    }

    JacobiSVD<MatrixXd> svd(jacobi_new, ComputeThinU | ComputeThinV);
    VectorXd x = svd.solve(v_new);

    return x;
}

VectorXd uGen3Ctrl::Gen3_QMgo(VectorXd Q, VectorXd v, double Ratio)
{
    MatrixXd jacobi = Gen3_Jacobi(Q);
    MatrixXd jacobi_null = Gen3_Null(jacobi);
    MatrixXd jacobi_new(jacobi.rows() + jacobi_null.rows(), jacobi.cols());
    jacobi_new << jacobi, jacobi_null.transpose();

    VectorXd v_new(v.size() + jacobi_null.rows());
    for (int i = 0; i < v.size(); i++)
    {
        v_new(i) = v[i];
    }
    for (int i = 0; i < jacobi_null.cols(); i++)
    {
        v_new(v.size() + i) = 0;
    }

    JacobiSVD<MatrixXd> svd(jacobi_new, ComputeThinU | ComputeThinV);
    VectorXd outq = svd.solve(v_new);

    double manipu = Gen3_Manipulability(Q);
    double mU = 0;
    if (manipu < QM_d1)
        mU = 1 / (3 * (exp(manipu) - 1) + 1);
    else if (manipu < QM_d2)
        mU = (QM_d2 - manipu) / (QM_d2 - QM_d1) * 0.601;

    VectorXd gradien = Gen3_MangGradient(Q, manipu);
    return outq;
}

double uGen3Ctrl::Gen3_ArmAngle(VectorXd Q)
{
    Gen3_UpBtoH(Q, 1);
    MatrixXd R0(3, 3), R1(3, 3);
    Vector3d p0 = Gen3_TMat[5].block<3, 1>(0, 3) - Gen3_TMat[1].block<3, 1>(0, 3);
    Vector3d p1 = Gen3_TMat[3].block<3, 1>(0, 3) - Gen3_TMat[1].block<3, 1>(0, 3);
    // Vector3d p2 = Gen3_TMat[3].block<3, 1>(0, 3);
    p0.normalize();
    Vector3d p_temp = -p0.cross(Vector3d(0, 0, 1));
    p_temp.normalize();
    R0 << p0, p_temp, p0.cross(p_temp);
    p_temp = -p0.cross(p1);
    p_temp.normalize();
    R1 << p0, p_temp, p0.cross(p_temp);
    Matrix3d Rw;
    Rw = R0 * R1.transpose();
    AngleAxisd rw(Rw);
    // rw.fromRotationMatrix(Rw);
    if (rw.axis().dot(p0) > 0)
        return -rw.angle();
    return rw.angle();
}

VectorXd uGen3Ctrl::Gen3_ArmAngleGradient(VectorXd Q, double armangle)
{
    double dangle = 1e-6;
    if (armangle > 95)
        armangle = Gen3_ArmAngle(Q);
    VectorXd darmangle(7, 1);
    for (int i = 0; i < 4; i++)
    {
        VectorXd dQ = Q;
        dQ[i] = dQ[i] + dangle;
        double armangle_ = Gen3_ArmAngle(dQ);
        darmangle(i) = (armangle_ - armangle) / dangle;
    }
    darmangle[4] = 0;
    darmangle[5] = 0;
    darmangle[6] = 0;
    return darmangle;
}

VectorXd uGen3Ctrl::Gen3_QAgo(VectorXd Q, VectorXd v, double minAngle, double maxAngle, double Ratio)
{
    MatrixXd jacobi = Gen3_Jacobi(Q);
    MatrixXd jacobi_null = Gen3_Null(jacobi);
    MatrixXd jacobi_new(jacobi.rows() + jacobi_null.cols(), jacobi.cols());
    jacobi_new << jacobi, jacobi_null.transpose();

    VectorXd v_new(v.size() + jacobi_null.rows());
    for (int i = 0; i < v.size(); i++)
    {
        v_new(i) = v[i];
    }
    for (int i = 0; i < jacobi_null.cols(); i++)
    {
        v_new(v.size() + i) = 0;
    }

    JacobiSVD<MatrixXd> svd(jacobi_new, ComputeThinU | ComputeThinV);
    VectorXd outq;
    outq = svd.solve(v_new);
    double armangle = Gen3_ArmAngle(Q);
    VectorXd aangleVec = Gen3_ArmAngleGradient(Q, armangle);

    if (armangle < minAngle)
    {
        armangle -= minAngle;
        VectorXd U0Vec = jacobi_null.transpose() * aangleVec;
        for (size_t i = 0; i < U0Vec.size(); i++)
        {
            VectorXd u0temp = Ratio * U0Vec[i] * jacobi_null.col(i) * atan(-armangle * 5);
            double dtemp = u0temp.cwiseAbs().maxCoeff();
            if (dtemp > 0.5)
                u0temp = u0temp / dtemp * 0.5;
            outq += u0temp;
        }
    }
    else if (armangle > maxAngle)
    {
        armangle -= maxAngle;
        VectorXd U0Vec = -jacobi_null.transpose() * aangleVec;
        for (size_t i = 0; i < U0Vec.size(); i++)
        {
            VectorXd u0temp = Ratio * U0Vec[i] * jacobi_null.col(i) * atan(armangle * 5);
            double dtemp = u0temp.cwiseAbs().maxCoeff();
            if (dtemp > 0.5)
                u0temp = u0temp / dtemp * 0.5;
            outq += u0temp;
        }
    }
    double dtemp = outq.cwiseAbs().maxCoeff();
    if (dtemp > 0.5)
        outq = outq / dtemp * 0.5;
    return outq;
}

// int main()
// {
//     uGen3Ctrl Gen3obj;
//     VectorXd Q(7), v(7);

//     Q << 35.162, 69.979, 116.032, 235.314, 90.973, 302.734, 2.53;
//     // Q << 40.751,47.643,132.653,224.033,74.46,325.789,21.25;
//     // Q << 329.3, 46.813, 234.378, 221.195, 286.251, 323.722, 174.091;
//     Q = Q / 180 * 3.1415926;
//     cout << Q << endl;
//     cout << "tool:\r\n"
//          << Gen3obj.Gen3_UpBtoH(Q) << endl;
//     double aangle = Gen3obj.Gen3_ArmAngle(Q);
//     printf("Gen3_ArmAngle: %f\r\n", aangle / 3.1415926 * 180);
//     cout << "Gen3_ArmAngleGradient:\r\n"
//          << Gen3obj.Gen3_ArmAngleGradient(Q) << endl;
//     v << 0, 0, 0.05, 0, 0, 0;
//     cout << "Gen3_QAgo:\r\n"
//          << Gen3obj.Gen3_QAgo(Q, v, 0) << endl;

//     cout << "Gen3_QAgo:\r\n"
//          << Gen3obj.Gen3_QAgo(Q, v, 1) << endl;

//     return 0;
// }
