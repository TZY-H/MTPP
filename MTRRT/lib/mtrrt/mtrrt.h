#ifndef __MTRRT_H
#define __MTRRT_H

#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <list>
#include <set>
#include <map>
// #include <thread>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "nanoflann.hpp"
#define statusDimension 2
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 2>>,
    nanoflann::PointCloud_d<double, 2>, 2>
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

// struct RRDT_ARM
// {
//     double P_success = 0.5;
//     int32_t treeIndex;
//     int32_t Count = 1;
//     int32_t successCount = 1;
//     status pos;
//     status direction;
//     uint8_t directionKey = false;
// };

struct PriorityEdge // 用于priority_queue
{
    double cost;
    status par; // 父节点
    status sub; // 子节点
    bool operator>(const PriorityEdge &other) const
    {
        return cost > other.cost;
    }
    bool operator<(const PriorityEdge &other) const
    {
        return cost < other.cost;
    }
};

class MTRRTplanner
{
private:
    double spatialExtentMin[statusDimension];
    double spatialExtentMax[statusDimension];
    double frontierMin, frontierMax;

    void Neighbor(const status &x, std::list<status> &Xnear, double R, kd_tree &Tree, nanoflann::PointCloud_d<double, statusDimension> &TreeCloud)
    {
        // double NN = TreeCloud.pts.size() + 1;
        // R = std::min(30.0, 1.1 * frontierMax * std::pow(log(NN) / NN, 1 / static_cast<double>(statusDimension)));
        Xnear.clear();
        // 构造查询点
        nanoflann::PointCloud_d<double, statusDimension>::Point query_pt;
        for (int32_t i = 0; i < statusDimension; i++)
            query_pt.q[i] = x.q[i];
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
            for (int32_t i = 0; i < statusDimension; i++)
                candidate.q[i] = point.q[i];
            Xnear.push_back(candidate);
        }
    }

    bool ValidityCheck(const status &x) //!!
    {
        if (Obsmap.at<uint8_t>(x.q[1], x.q[0]))
            return false;
        else
            return true;
    }
    double HeuristicCostFun(const status &S, const status &E)
    {
        double sum = 0;
        for (int32_t i = 0; i < statusDimension; i++)
        {
            double dx = S.q[i] - E.q[i];
            sum += (dx * dx);
        }
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
                double x = S.q[0] * t + (1 - t) * E.q[0];
                double y = S.q[1] * t + (1 - t) * E.q[1];
                if (!ValidityCheck({x, y}))
                    return doubleMAX;
            }
        }
        return realCost;
    }

    bool SampleStatus(status &x_new)
    {
        // static std::mt19937 gen(rd);
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
                   nanoflann::PointCloud_d<double, statusDimension> &TreeCloud)
    {
        nanoflann::PointCloud_d<double, statusDimension>::Point query_pt;
        for (int32_t i = 0; i < statusDimension; i++)
            query_pt.q[i] = x.q[i];
        TreeCloud.pts.push_back(query_pt);
        Tree.addPoints(TreeCloud.pts.size() - 1, TreeCloud.pts.size() - 1);
    }

    status Extend(const status &S, const status &E)
    {
        bool key = false;
        double len = HeuristicCostFun(S, E);
        int32_t MaximumDiscrete = len / 4 + 2; //!!
        status xold = S;
        for (int32_t i = 1; i < MaximumDiscrete; i++)
        {
            double t = (double)i / MaximumDiscrete;
            double x = E.q[0] * t + (1 - t) * S.q[0];
            double y = E.q[1] * t + (1 - t) * S.q[1];
            status xnow = {x, y};
            if (!ValidityCheck(xnow))
                return xold;
            xold = xnow;
        }
        return E;
    }

    void MergeTree(status &x_1, status &x_2, std::map<int32_t, status> &treeHead, std::map<status, Node> &NodeMap)
    {
        status x_old = x_1;
        status x_now = x_2;
        NodeMap[x_old].subs.insert(x_now);
        while (!(NodeMap[x_now].head))
        {
            Node &node_now = NodeMap[x_now];
            status x_par = node_now.par;

            node_now.par = x_old;
            node_now.subs.insert(x_par);
            node_now.subs.erase(x_old);

            x_old = x_now;
            x_now = x_par;
        }
        Node &node_now = NodeMap[x_now];
        node_now.head = false;
        node_now.par = x_old;
        node_now.subs.erase(x_old);

        treeHead.erase(NodeMap[x_2].tree);
        int32_t treeIndexNew = NodeMap[x_1].tree;
        std::list<status> Qup;
        Qup.push_back(x_2);
        while (Qup.size())
        {
            Node &node_now = NodeMap[Qup.front()];
            node_now.tree = treeIndexNew;
            Qup.insert(Qup.end(), node_now.subs.begin(), node_now.subs.end());
            Qup.pop_front();
        }
    }

    void getTreeNode(status treeRoot, std::set<status> &Xset, std::map<status, Node> &NodeMap)
    {
        std::list<status> Qup;
        Qup.push_back(treeRoot);
        Xset.insert(treeRoot);
        while (Qup.size())
        {
            Node &node_now = NodeMap[Qup.front()];
            Qup.insert(Qup.end(), node_now.subs.begin(), node_now.subs.end());
            Xset.insert(node_now.subs.begin(), node_now.subs.end());
            Qup.pop_front();
        }
    }

    void Rewiring(std::set<status> &Xnew, std::map<status, Node> &NodeMap, kd_tree &Tree,
                  nanoflann::PointCloud_d<double, statusDimension> &TreeCloud)
    {
        std::priority_queue<PriorityEdge,
                            std::vector<PriorityEdge>,
                            std::greater<PriorityEdge>>
            Qs;
        for (const status &x_new : Xnew)
        {
            std::list<status> Xnear;
            Neighbor(x_new, Xnear, 1.25 * lambda, Tree, TreeCloud);
            for (const status &xnear : Xnear)
            {
                Node &node_near = NodeMap[xnear];
                if (node_near.tree != 0 || xnear == x_new)
                    continue;
                double costH = node_near.cost + HeuristicCostFun(x_new, xnear);
                if (NodeMap[x_new].cost > costH)
                    Qs.push({costH, xnear, x_new});
            }
        }

        while (Qs.size())
        {
            double costH = Qs.top().cost;
            status x_par = Qs.top().par;
            status x_new = Qs.top().sub;
            Qs.pop();
            Node &node_par = NodeMap[x_par];
            Node &node_new = NodeMap[x_new];
            if (costH < node_new.cost)
            {
                double costR = node_par.cost + RealCostFun(x_par, x_new);
                if (costR < node_new.cost)
                {
                    node_new.cost = costR;
                    NodeMap[node_new.par].subs.erase(x_new);
                    node_new.subs.insert(x_new);
                    node_new.par = x_par;

                    std::list<status> Xnear;
                    Neighbor(x_new, Xnear, 1.25 * lambda, Tree, TreeCloud);
                    for (const status &xnear : Xnear)
                    {
                        Node &node_near = NodeMap[xnear];
                        if (node_near.tree != 0 || xnear == x_new)
                            continue;
                        double costH = node_new.cost + HeuristicCostFun(x_new, xnear);
                        if (node_near.cost > costH)
                            Qs.push({costH, x_new, xnear});
                    }
                }
            }
        }
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
    double lambda = 50;
    long t_init, t_opt;
    double c_init;

    // int32_t shapeX;  // 地图x轴像素数
    // int32_t shapeY;  // 地图y轴像素数
    cv::Mat Freemap; // free图象显示
    cv::Mat Obsmap;  // obs图象碰撞检测

    MTRRTplanner(int32_t rd) : gen(rd) {}
    void InitializeInvironment(const char *IMGmap, int th)
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
        spatialExtentMax[0] = Obsmap.cols;
        spatialExtentMax[1] = Obsmap.rows;
        frontierMin = std::min(Obsmap.cols, Obsmap.rows);
        frontierMax = std::max(Obsmap.cols, Obsmap.rows);
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
        c_init = 0;
        t_init = t_opt = 0;
        long t0 = utime_ns();
        path.clear();
        nanoflann::PointCloud_d<double, statusDimension> NeighborTreeCloud;
        kd_tree NeighborTree(statusDimension, NeighborTreeCloud,
                             nanoflann::KDTreeSingleIndexAdaptorParams(10));
        std::map<status, Node> NodeMap;
        std::map<int32_t, status> treeHead;

        int32_t treeIndexCount = 0;
        // 初始化n个tree
        NodeMap[init] = {0, treeIndexCount, true, {}, {}};
        treeHead[treeIndexCount] = init;
        addKDtree(init, NeighborTree, NeighborTreeCloud);
        treeIndexCount++;
        { // 添加goal subtree
            NodeMap[goal] = {doubleMAX, treeIndexCount, true, {}, {}};
            treeHead[treeIndexCount] = goal;
            addKDtree(goal, NeighborTree, NeighborTreeCloud);
            treeIndexCount++;
        }
        for (int32_t armIndex = 1; armIndex < armNum; armIndex++)
        {
            status x_new;
            if (!SampleStatus(x_new))
                continue;
            NodeMap[x_new] = {doubleMAX, treeIndexCount, true, {}, {}};
            treeHead[treeIndexCount] = x_new;
            addKDtree(x_new, NeighborTree, NeighborTreeCloud);
            treeIndexCount++;
        }

        int32_t n = 0;
        std::list<std::pair<status, status>> C_Tree; // 弹出时需判断树是否已被统一
        while (n <= N)
        {
            if (!C_Tree.size())
            {
                status x_rand;
                if (!SampleStatus(x_rand))
                    continue;
                bool DistKey = false;
                bool DistKey_root = false;
                status x_near_info;
                status x_near_root;
                {
                    std::list<status> Xnear;
                    Neighbor(x_rand, Xnear, lambda, NeighborTree, NeighborTreeCloud);
                    if (Xnear.size())
                    {
                        DistKey = true;
                        x_near_info = Xnear.front();
                    }
                    for (status &x_near : Xnear)
                    {
                        if (NodeMap[x_near].tree == 0)
                        {
                            DistKey_root = true;
                            x_near_root = x_near;
                            break;
                        }
                    }
                }
                if (DistKey_root)
                {
                    status x_new = Extend(x_near_root, x_rand);
                    if (x_new == x_near_root)
                        continue;
                    Node &node_par = NodeMap[x_near_root];
                    node_par.subs.insert(x_new);
                    NodeMap[x_new] = {node_par.cost + HeuristicCostFun(x_near_root, x_rand), 0, false, x_near_root, {}};
                    addKDtree(x_new, NeighborTree, NeighborTreeCloud);

                    std::set<status> Xnew;
                    Xnew.insert(x_new);
                    Rewiring(Xnew, NodeMap, NeighborTree, NeighborTreeCloud);

                    std::list<status> Xnear;
                    Neighbor(x_new, Xnear, lambda, NeighborTree, NeighborTreeCloud);
                    for (status &x_near : Xnear)
                    {
                        if (NodeMap[x_near].tree != 0)
                        {
                            double realCost = RealCostFun(x_new, x_near);
                            if (realCost < doubleMAX)
                                C_Tree.push_back({x_new, x_near});
                        }
                    }
                    n++;
                }
                else if (DistKey)
                {
                    double realCost = RealCostFun(x_near_info, x_rand);
                    if (realCost < doubleMAX)
                    {
                        Node &node_par = NodeMap[x_near_info];
                        node_par.subs.insert(x_rand);
                        int32_t treeIndex = node_par.tree;
                        NodeMap[x_rand] = {doubleMAX, treeIndex, false, x_near_info, {}};
                        addKDtree(x_rand, NeighborTree, NeighborTreeCloud);

                        std::list<status> Xnear;
                        Neighbor(x_rand, Xnear, lambda, NeighborTree, NeighborTreeCloud);
                        for (status &x_near : Xnear)
                        {
                            if (NodeMap[x_near].tree != treeIndex)
                            {
                                double realCost = RealCostFun(x_rand, x_near);
                                if (realCost < doubleMAX)
                                    C_Tree.push_back({x_rand, x_near});
                            }
                        }
                        n++;
                    }
                }
                else
                {
                    NodeMap[x_rand] = {doubleMAX, treeIndexCount, true, {}, {}};
                    treeHead[treeIndexCount] = x_rand;
                    addKDtree(x_rand, NeighborTree, NeighborTreeCloud);
                    treeIndexCount++;
                    n++;
                }
            }

            if (C_Tree.size())
            {
                status x_1 = C_Tree.front().first;
                status x_2 = C_Tree.front().second;
                Node &node1 = NodeMap[x_1];
                Node &node2 = NodeMap[x_2];
                C_Tree.pop_front();
                if (node1.tree == node2.tree)
                    continue;
                if (node1.tree == 0)
                {
                    std::set<status> Xnew;
                    getTreeNode(treeHead[node2.tree], Xnew, NodeMap);
                    MergeTree(x_1, x_2, treeHead, NodeMap);
                    Rewiring(Xnew, NodeMap, NeighborTree, NeighborTreeCloud); // 修改
                }
                else
                {
                    double realCost = RealCostFun(x_1, x_2);
                    if (realCost < doubleMAX)
                    {
                        MergeTree(x_1, x_2, treeHead, NodeMap);
                    }
                }
            }
            if (NodeMap.count(goal))
            {
                if (t_init == 0 && NodeMap[goal].cost < doubleMAX)
                {
                    t_init = utime_ns() - t0;
                    c_init = NodeMap[goal].cost;
                }
                if (NodeMap[goal].cost <= allowCost)
                    break;
            }
        }
        double cost = getpath(init, goal, NodeMap, path);
        t_opt = utime_ns() - t0;
        return cost;
        // if (n % 50 == 0 && n >= 780)
        // {
        //     printf("n! %d\r\n", n);
        //     printf("N! %ld\r\n", NodeMap.size());

        //     cv::Mat showMap = Freemap.clone();
        //     for (const auto nodepair : NodeMap)
        //     {
        //         const status &pnow = nodepair.first;
        //         cv::Point endPoint((int)pnow.q[0], (int)pnow.q[1]);
        //         if (nodepair.second.tree == 70)
        //             cv::circle(showMap, endPoint, 3, cv::Scalar(0, 0, 255), cv::FILLED);
        //         else
        //             cv::circle(showMap, endPoint, 3, cv::Scalar(0, 255, 0), cv::FILLED);

        //         if (nodepair.second.head)
        //             continue;
        //         const status &ppar = nodepair.second.par;
        //         if (ppar.q[0] < 0)
        //             continue;
        //         cv::Point startPoint((int)ppar.q[0], (int)ppar.q[1]);
        //         cv::line(showMap, startPoint, endPoint, cv::Scalar(255, 0, 0), 1);
        //         if (nodepair.second.cost >= doubleMAX)
        //             cv::circle(showMap, startPoint, 2, cv::Scalar(255, 0, 0), cv::FILLED);
        //     }

        //     cv::Point PointS((int)init.q[0], (int)init.q[1]);
        //     cv::circle(showMap, PointS, 2, cv::Scalar(255, 255, 0), cv::FILLED);
        //     cv::Point PointG((int)goal.q[0], (int)goal.q[1]);
        //     cv::circle(showMap, PointG, 2, cv::Scalar(255, 255, 0), cv::FILLED);

        //     std::list<status> path;
        //     double cost = getpath(init, goal, NodeMap, path);
        //     if (path.size() >= 2)
        //     {
        //         status point0 = path.front();
        //         for (status point1 : path)
        //         {
        //             cv::Point startPoint((int)point0.q[0], (int)point0.q[1]);
        //             cv::Point endPoint((int)point1.q[0], (int)point1.q[1]);
        //             cv::line(showMap, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);
        //             point0 = point1;
        //         }
        //     }

        //     printf("cost:%lf\r\n", cost);
        //     cv::imshow("showMap", showMap);
        //     cv::waitKey(0);
        // }
        // }
    }
};

long long utime_ns(void);

#endif
