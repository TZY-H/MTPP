#ifndef __RRF_H
#define __RRF_H

#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <thread>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "nanoflann.hpp"
#define statusDimension 3
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 3>>,
    nanoflann::PointCloud_d<double, 3>, 3>
    kd_tree;

#define doubleMAX (1.79769e308)

void generateRandomUnitVector(int n, std::vector<double> &random_point, std::mt19937 &gen);
void vonMisesFisher(int n, const std::vector<double> &vec_O, double kappa, std::vector<double> &vec_new, std::mt19937 &gen);
double fast_vonmises(double mu, double kappa, std::mt19937 &gen);

struct status
{
    double q[statusDimension];

    bool operator==(const status &other) const
    {
        for (int32_t i = 0; i < statusDimension; i++)
        {
            if (std::abs(q[i] - other.q[i]) >= 1e-10)
                return false;
        }
        return true;
    }
    bool operator!=(const status &other) const
    {
        for (int32_t i = 0; i < statusDimension; i++)
        {
            if (std::abs(q[i] - other.q[i]) >= 1e-10)
                return true;
        }
        return false;
    }
    bool operator<(const status &other) const
    {
        for (int32_t i = 0; i < statusDimension; i++)
        {
            // 使用与 operator== 一致的容差条件判断是否“不相等”
            if (std::abs(q[i] - other.q[i]) >= 1e-10)
                return q[i] < other.q[i];
        }
        return false;
    }
};

struct Node
{
    double cost = doubleMAX;
    int32_t tree;
    uint8_t head;
    status par;
    std::set<status> subs;
};

struct RRF_ARM
{
    double P_success = 0.5;
    int32_t treeIndex;
    int32_t Count = 1;
    int32_t successCount = 1;
    status pos;
    status direction;
    uint8_t directionKey = false;
};

class RRFplanner
{
private:
    double spatialExtentMin[statusDimension];
    double spatialExtentMax[statusDimension];
    double frontierMin, frontierMax;
    double RotationCoefficient;

    void Neighbor(const status &x, std::list<status> &Xnear, double R, kd_tree &Tree, nanoflann::PointCloud_d<double, statusDimension> &TreeCloud) //!!
    {
        Xnear.clear();
        // 构造查询点
        nanoflann::PointCloud_d<double, statusDimension>::Point query_pt;
        // for (int32_t i = 0; i < statusDimension; i++)
        //     query_pt.q[i] = x.q[i];
        query_pt.q[0] = x.q[0];
        query_pt.q[1] = x.q[1];
        query_pt.q[2] = RotationCoefficient * x.q[2];

        // 存储结果的容器
        std::vector<nanoflann::ResultItem<uint32_t, double>> IndicesDists;
        // 半径参数需要传入距离的平方
        nanoflann::RadiusResultSet<double, uint32_t> resultSet(R * R, IndicesDists);
        // 执行范围查询
        Tree.findNeighbors(resultSet, query_pt.q);
        // 遍历结果并排除自身点
        for (const auto &result : IndicesDists)
        {
            const auto &point = TreeCloud.pts[result.first];
            status candidate;
            // for (int32_t i = 0; i < statusDimension; i++)
            candidate.q[0] = point.q[0];
            candidate.q[1] = point.q[1];
            candidate.q[2] = point.q[2] / RotationCoefficient;
            Xnear.push_back(candidate);
        }
    }

    bool ValidityCheck(const status &x) //!!
    {
        // objectPointSet
        double costh = cos(x.q[2]);
        double sinth = sin(x.q[2]);
        for (const status &p : objectPointSet)
        {
            double px = costh * p.q[0] - sinth * p.q[1] + x.q[0];
            double py = sinth * p.q[0] + costh * p.q[1] + x.q[1];
            if (px < spatialExtentMin[0] || px > spatialExtentMax[0] || py < spatialExtentMin[1] || py > spatialExtentMax[1])
                return false;
            if (Obsmap.at<uint8_t>(py, px))
                return false;
        }
        return true;
    }

    double HeuristicCostFun(const status &S, const status &E) //!!
    {
        double sum = 0;
        double dx = S.q[0] - E.q[0];
        double dy = S.q[1] - E.q[1];
        double dr = RotationCoefficient * (S.q[2] - E.q[2]);
        sum += (dx * dx + dy * dy + dr * dr);
        return sqrt(sum);
    }

    double RealCostFun(const status &S, const status &E)
    {
        double realCost = HeuristicCostFun(S, E);

        if (!(ValidityCheck(S) && ValidityCheck(E)))
            return doubleMAX;

        bool key = false;
        double len = realCost;
        int32_t MaximumDiscrete = len / 4 + 2; //!!
        for (double denominator = 2; denominator <= MaximumDiscrete; denominator *= 2)
        {
            for (int32_t i = 1; i < denominator; i += 2)
            {
                double t = i / denominator;
                status xnow;
                for (int32_t j = 0; j < statusDimension; j++)
                    xnow.q[j] = E.q[j] * t + (1 - t) * S.q[j];

                if (!ValidityCheck(xnow))
                    return doubleMAX;
            }
        }
        return realCost;
    }

    bool SampleStatus(status &x_new)
    {
        std::uniform_real_distribution<double> distribI(0, 1);
        Node sampleNode = {doubleMAX, -1, {}, {}};

        int32_t sampleNumMax = 20000;

        while (sampleNumMax-- > 0)
        {
            status x_sam;
            for (int32_t i = 0; i < statusDimension; i++)
                x_sam.q[i] = (spatialExtentMax[i] - spatialExtentMin[i]) * distribI(gen) + spatialExtentMin[i];
            if (!ValidityCheck(x_sam))
                continue;
            x_new = x_sam;
            return true;
        }
        return false;
    }

    void addKDtree(status x, kd_tree &Tree,
                   nanoflann::PointCloud_d<double, statusDimension> &TreeCloud) //!!
    {
        nanoflann::PointCloud_d<double, statusDimension>::Point query_pt;
        query_pt.q[0] = x.q[0];
        query_pt.q[1] = x.q[1];
        query_pt.q[2] = RotationCoefficient * x.q[2];
        TreeCloud.pts.push_back(query_pt);
        Tree.addPoints(TreeCloud.pts.size() - 1, TreeCloud.pts.size() - 1);
    }

    long long utime_ns(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<long long>(1000000000UL) * static_cast<long long>(ts.tv_sec) +
               static_cast<long long>(ts.tv_nsec);
    }

public:
    std::mt19937 gen;
    int32_t armNum = 50;
    double PolynomialThreshold = 0.05;
    double E_distance = 50;
    long t_init, t_opt;
    double c_init;

    int32_t treeIndexCount = 0;

    std::vector<status> objectPointSet;

    cv::Mat Freemap; // free图象显示
    cv::Mat Obsmap;  // obs图象碰撞检测

    RRFplanner(int32_t rd) : gen(rd) {}
    void InitializeInvironment(const char *IMGmap, int th, double radian = 15.7079632)
    {
        // 读取输入的图像
        cv::Mat inputImage = cv::imread(IMGmap, cv::IMREAD_GRAYSCALE);

        if (inputImage.empty())
        {
            std::cerr << "Failed to load input image." << std::endl;
            return;
        }

        // 对图像进行二值化处理
        cv::threshold(inputImage, Freemap, th, 255, cv::THRESH_BINARY);

        // 确保输出矩阵Freemap的数据类型为CV_8U
        if (Freemap.type() != CV_8U)
        {
            Freemap.convertTo(Freemap, CV_8U);
        }
        cv::bitwise_not(Freemap, Obsmap);
        Freemap = cv::imread(IMGmap);
        spatialExtentMin[0] = 0;
        spatialExtentMin[1] = 0;
        spatialExtentMin[2] = -radian;
        spatialExtentMax[0] = Obsmap.cols;
        spatialExtentMax[1] = Obsmap.rows;
        spatialExtentMax[2] = radian;
        frontierMin = std::min(Obsmap.cols, Obsmap.rows);
        frontierMax = std::max(Obsmap.cols, Obsmap.rows);
        RotationCoefficient = frontierMax / radian / 2.0;
    }

    bool RestartArm(std::vector<RRF_ARM> &armSet, std::map<status, Node> &NodeMap, kd_tree &Tree,
                    nanoflann::PointCloud_d<double, statusDimension> &TreeCloud)
    {
        for (RRF_ARM &arm : armSet)
        {
            if (arm.P_success < PolynomialThreshold)
            {
                status x_new;
                if (!SampleStatus(x_new))
                    continue;

                status xlink_opt;
                bool linkRoot = false;
                bool JoinKey = false;
                double minCost = doubleMAX;
                std::list<status> Xnear;
                Neighbor(x_new, Xnear, E_distance, Tree, TreeCloud);
                for (status &x_near : Xnear)
                {
                    if (x_new == x_near)
                        continue;
                    Node &node_par = NodeMap[x_near];
                    int32_t par_tree = node_par.tree;
                    if (par_tree == arm.treeIndex)
                        continue;
                    double realCost = RealCostFun(x_near, x_new);
                    if (realCost >= doubleMAX)
                        continue;
                    if (linkRoot)
                    {
                        if (par_tree == 0)
                        {
                            double newCost = node_par.cost + realCost;
                            if (minCost > newCost)
                            {
                                minCost = newCost;
                                xlink_opt = x_near;
                            }
                        }
                    }
                    else
                    {
                        JoinKey = true;
                        if (par_tree == 0)
                        {
                            linkRoot = true;
                            xlink_opt = x_near;
                            minCost = node_par.cost + realCost;
                        }
                        else if (minCost > realCost)
                        {
                            xlink_opt = x_near;
                            minCost = realCost;
                        }
                    }
                }

                if (JoinKey)
                {
                    Node &node_par = NodeMap[xlink_opt];
                    node_par.subs.insert(x_new);
                    Node &node_xnew = NodeMap[x_new];
                    node_xnew = {doubleMAX, node_par.tree, false, xlink_opt, {}};
                    addKDtree(x_new, Tree, TreeCloud);

                    if (node_par.tree == 0)
                    {
                        node_xnew.cost = minCost;
                        std::list<status> Qr;
                        Qr.push_back(x_new);
                        while (Qr.size())
                        {
                            status x_rnow = Qr.front();
                            Qr.pop_front();
                            Node &node_rnow = NodeMap[x_rnow];

                            std::list<status> Xnear_r;
                            Neighbor(x_rnow, Xnear_r, 1.25 * E_distance, Tree, TreeCloud);
                            for (const status &xnear_r : Xnear_r)
                            {
                                Node &node_nearr = NodeMap[xnear_r];
                                if (x_rnow == xnear_r)
                                    continue;
                                double c_new = RealCostFun(xnear_r, x_rnow);
                                if (c_new >= doubleMAX)
                                    continue;
                                c_new += node_rnow.cost;
                                if (c_new < node_nearr.cost)
                                {
                                    if (!node_nearr.head)
                                        NodeMap[node_nearr.par].subs.erase(xnear_r);
                                    node_rnow.subs.insert(xnear_r);
                                    node_nearr.par = x_rnow;
                                    node_nearr.head = false;
                                    node_nearr.tree = 0;
                                    node_nearr.cost = c_new;
                                    Qr.push_back(xnear_r);
                                }
                            }
                        }
                    }
                }
                else
                {
                    arm.P_success = 0.5;
                    arm.treeIndex = treeIndexCount;
                    arm.Count = 0;
                    arm.successCount = 0;
                    arm.pos = x_new;
                    arm.directionKey = false;

                    NodeMap[x_new] = {doubleMAX, treeIndexCount, true, {}, {}};
                    addKDtree(x_new, Tree, TreeCloud);
                    treeIndexCount++;
                }
                return true;
            }
        }

        return false;
    }

    double getpath(status init, status goal, std::map<status, Node> &NodeMap, std::list<status> &path)
    {
        path.resize(0);
        if (!NodeMap.count(goal))
            return -1;
        if (NodeMap[goal].cost >= doubleMAX)
            return -1;
        status x_now = goal;
        while (init != x_now)
        {
            path.push_front(x_now);
            x_now = NodeMap[x_now].par;
        }
        path.push_front(init);
        return NodeMap[goal].cost;
    }

    double planningTask(status init, status goal, std::list<status> &path, double allowCost = 0.0, int32_t N = 10000)
    {
        path.clear();
        c_init = 0;
        treeIndexCount = 0;
        t_init = t_opt = 0;
        long t0 = utime_ns();

        std::uniform_real_distribution<double> distribI(0, 1);

        nanoflann::PointCloud_d<double, statusDimension> NeighborSearchTreeCloud;
        kd_tree NeighborSearchTree(statusDimension, NeighborSearchTreeCloud,
                                   nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::map<status, Node> NodeMap;
        NodeMap[init] = {0, treeIndexCount, true, {}, {}};
        addKDtree(init, NeighborSearchTree, NeighborSearchTreeCloud);
        treeIndexCount++;

        // 初始化k个ARM
        std::vector<RRF_ARM> armSet;
        { // 添加goal subtree
            RRF_ARM arm_new;
            arm_new.pos = goal;
            arm_new.treeIndex = treeIndexCount;
            armSet.push_back(arm_new);
            NodeMap[goal] = {doubleMAX, treeIndexCount, true, {}, {}};
            addKDtree(goal, NeighborSearchTree, NeighborSearchTreeCloud);
            treeIndexCount++;
        }
        for (int32_t armIndex = 1; armIndex < armNum; armIndex++)
        {
            status x_new;
            if (!SampleStatus(x_new))
                continue;
            RRF_ARM arm_new;
            arm_new.pos = x_new;
            arm_new.treeIndex = treeIndexCount;
            armSet.push_back(arm_new);
            NodeMap[x_new] = {doubleMAX, treeIndexCount, true, {}, {}};
            addKDtree(x_new, NeighborSearchTree, NeighborSearchTreeCloud);
            treeIndexCount++;
        }

        // cv::Mat showMap = Freemap.clone();
        // for (auto &armPair:armSet)
        // {
        //     showOneC(armPair.pos,showMap);
        // }
        // cv::imshow("showMap",showMap);
        // cv::waitKey(30);

        int32_t n = 0;
        while (n <= N)
        {
            if (RestartArm(armSet, NodeMap, NeighborSearchTree, NeighborSearchTreeCloud))
                n++;
            else
            {
                double sum = 0;
                for (RRF_ARM &arm : armSet)
                    sum += arm.P_success;
                if (sum == 0)
                    continue;

                double p = sum * distribI(gen);
                sum = 0;
                int32_t armIndex = 0;
                for (RRF_ARM &arm : armSet)
                {
                    sum += arm.P_success;
                    if (sum > p)
                        break;
                    armIndex++;
                }
                RRF_ARM &arm_i = armSet[armIndex];
                std::vector<double> x_new_direction(statusDimension);
                if (arm_i.directionKey)
                {
                    std::vector<double> armi_direction(statusDimension);
                    for (int32_t i = 0; i < statusDimension; i++)
                        armi_direction[i] = arm_i.direction.q[i];
                    vonMisesFisher(statusDimension, armi_direction, 1, x_new_direction, gen);
                }
                else
                    generateRandomUnitVector(statusDimension, x_new_direction, gen);
                status x_new;
                for (int32_t i = 0; i < statusDimension; i++)
                    x_new.q[i] = E_distance * x_new_direction[i] + arm_i.pos.q[i];
                arm_i.Count++;

                double realCost_ai2new = RealCostFun(arm_i.pos, x_new);
                if (realCost_ai2new < doubleMAX)
                {
                    n++;
                    Node &node_ai = NodeMap[arm_i.pos];
                    Node &node_xnew = NodeMap[x_new];
                    node_ai.par = x_new;
                    node_ai.head = false;
                    node_xnew.head = true;
                    node_xnew.subs.insert(arm_i.pos);
                    node_xnew.cost = doubleMAX;
                    node_xnew.tree = arm_i.treeIndex;
                    addKDtree(x_new, NeighborSearchTree, NeighborSearchTreeCloud);
                    arm_i.directionKey = true;
                    double sumdirection = 0;
                    std::vector<double> direction(statusDimension);
                    for (int32_t i = 0; i < statusDimension; i++)
                    {
                        direction[i] = x_new.q[i] - arm_i.pos.q[i];
                        sumdirection += (direction[i] * direction[i]);
                    }
                    sumdirection = sqrt(sumdirection);
                    for (int32_t i = 0; i < statusDimension; i++)
                        arm_i.direction.q[i] = direction[i] / sumdirection;

                    arm_i.pos = x_new;
                    arm_i.successCount++;

                    status xlink_opt;
                    bool linkRoot = false;
                    double minCost = doubleMAX;
                    std::list<status> Xnear;
                    Neighbor(x_new, Xnear, E_distance, NeighborSearchTree, NeighborSearchTreeCloud);
                    for (status &x_near : Xnear)
                    {
                        Node &node_par = NodeMap[x_near];
                        int32_t par_tree = node_par.tree;
                        if (par_tree == arm_i.treeIndex)
                            continue;
                        double realCost = RealCostFun(x_near, x_new);
                        if (realCost >= doubleMAX)
                            continue;
                        if (linkRoot)
                        {
                            if (par_tree == 0)
                            {
                                double newCost = node_par.cost + realCost;
                                if (minCost > newCost)
                                {
                                    minCost = newCost;
                                    xlink_opt = x_near;
                                }
                            }
                        }
                        else
                        {
                            if (par_tree == 0)
                            {
                                linkRoot = true;
                                xlink_opt = x_near;
                                minCost = node_par.cost + realCost;
                            }
                            else if (minCost > realCost)
                            {
                                xlink_opt = x_near;
                                minCost = realCost;
                            }
                        }
                    }

                    if (minCost < doubleMAX)
                    {
                        Node &node_par = NodeMap[xlink_opt];
                        int32_t par_tree = node_par.tree;

                        // 子树合并
                        arm_i.P_success = 0;
                        arm_i.treeIndex = -1;

                        node_par.subs.insert(x_new);
                        node_xnew.par = xlink_opt;
                        node_xnew.head = false;

                        if (linkRoot)
                        {
                            node_xnew.cost = minCost;
                            std::list<status> Qr;
                            Qr.push_back(x_new);
                            while (Qr.size())
                            {
                                status x_rnow = Qr.front();
                                Qr.pop_front();
                                Node &node_rnow = NodeMap[x_rnow];

                                std::list<status> Xnear_r;
                                Neighbor(x_rnow, Xnear_r, 1.25 * E_distance, NeighborSearchTree, NeighborSearchTreeCloud);
                                for (const status &xnear_r : Xnear_r)
                                {
                                    Node &node_nearr = NodeMap[xnear_r];
                                    if (x_rnow == xnear_r)
                                        continue;
                                    double c_new = RealCostFun(xnear_r, x_rnow);
                                    if (c_new >= doubleMAX)
                                        continue;
                                    c_new += node_rnow.cost;
                                    if (c_new < node_nearr.cost)
                                    {
                                        if (!node_nearr.head)
                                            NodeMap[node_nearr.par].subs.erase(xnear_r);
                                        node_rnow.subs.insert(xnear_r);
                                        node_nearr.par = x_rnow;
                                        node_nearr.tree = 0;
                                        node_nearr.head = false;
                                        node_nearr.cost = c_new;
                                        Qr.push_back(xnear_r);
                                    }
                                }
                            }
                        }
                    }
                }
                if (arm_i.treeIndex == -1)
                    arm_i.P_success = 0;
                else
                    arm_i.P_success = arm_i.successCount / arm_i.Count;
            }
            if (NodeMap.count(goal))
            {
                if (t_init == 0 && NodeMap[goal].cost < doubleMAX)
                {
                    t_init = utime_ns() - t0;
                    c_init = NodeMap[goal].cost;
                    printf("!! %lf\r\n", NodeMap[goal].cost);
                }
                if (NodeMap[goal].cost <= allowCost)
                    break;
            }
        }
        double cost = getpath(init, goal, NodeMap, path);
        t_opt = utime_ns() - t0;
        return cost;
    }
    void showPath(std::list<status> &Path, cv::Mat &showImg)
    {
        std::list<status> PathShow;
        PathShow.push_back(Path.front());
        status p_old = Path.front();
        double discreteDistance = 4.0;
        for (const status &p_now : Path)
        {
            double dx = p_old.q[0] - p_now.q[0];
            double dy = p_old.q[1] - p_now.q[1];
            double dr = RotationCoefficient * (p_old.q[2] - p_now.q[2]);
            double realCost = sqrt(dx * dx + dy * dy + dr * dr);
            double len = realCost;

            int32_t MaximumDiscrete = len / 4 + 2; //$$$
            for (int32_t i = 1; i <= MaximumDiscrete; i++)
            {

                double t = (double)i / MaximumDiscrete;
                double x = p_now.q[0] * t + (1 - t) * p_old.q[0];
                double y = p_now.q[1] * t + (1 - t) * p_old.q[1];
                double r = p_now.q[2] * t + (1 - t) * p_old.q[2];
                status pp = {x, y, r};
                if (HeuristicCostFun(pp, PathShow.back()) > 12.0)
                    PathShow.push_back(pp);
            }
            p_old = p_now;
        }
        PathShow.push_back(p_old);
        for (const status &pathP : PathShow)
        {
            cv::Mat showMap = showImg.clone();
            double costh = cos(pathP.q[2]);
            double sinth = sin(pathP.q[2]);
            double px_old = costh * objectPointSet.back().q[0] - sinth * objectPointSet.back().q[1] + pathP.q[0];
            double py_old = sinth * objectPointSet.back().q[0] + costh * objectPointSet.back().q[1] + pathP.q[1];
            for (const status &p : objectPointSet)
            {
                double px = costh * p.q[0] - sinth * p.q[1] + pathP.q[0];
                double py = sinth * p.q[0] + costh * p.q[1] + pathP.q[1];
                cv::Point pt1(static_cast<int>(px), static_cast<int>(py));
                // cv::circle(showMap, pt1, 3, cv::Scalar(255, 0, 0), -1);
                cv::Point pt2(static_cast<int>(px_old), static_cast<int>(py_old));
                cv::line(showMap, pt1, pt2, cv::Scalar(255, 0, 0), 3);
                px_old = px;
                py_old = py;
            }
            cv::imshow("showMap", showMap);
            cv::waitKey(60);
        }
    }

    void showOneC(status &x, cv::Mat &showImg)
    {
        double costh = cos(x.q[2]);
        double sinth = sin(x.q[2]);
        cv::Scalar color;
        if (ValidityCheck(x))
        {
            double blue = 100 + rand() % 128;
            double green = 100 + rand() % 128;
            double red = 80 + rand() % 128;
            color = cv::Scalar(blue, green, red);
        }
        else
            color = cv::Scalar(0, 0, 255);
        double px_old = costh * objectPointSet.back().q[0] - sinth * objectPointSet.back().q[1] + x.q[0];
        double py_old = sinth * objectPointSet.back().q[0] + costh * objectPointSet.back().q[1] + x.q[1];
        for (const status &p : objectPointSet)
        {
            double px = costh * p.q[0] - sinth * p.q[1] + x.q[0];
            double py = sinth * p.q[0] + costh * p.q[1] + x.q[1];
            cv::Point pt1(static_cast<int>(px), static_cast<int>(py));
            // cv::circle(Freemap, pt1, 3, color, -1);
            cv::Point pt2(static_cast<int>(px_old), static_cast<int>(py_old));
            cv::line(showImg, pt1, pt2, color, 3);
            px_old = px;
            py_old = py;
        }
    }
};

long long utime_ns(void);

#endif
