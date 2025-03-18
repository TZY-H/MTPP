#ifndef __PMTG7DOF_H
#define __PMTG7DOF_H
#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <list>
#include <thread>
#include <random>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <functional>
#include <cmath>

// #include <tbb/concurrent_priority_queue.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>

#include "nanoflann.hpp"
#include "uGen3Ctrl.h"

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 3>>,
    nanoflann::PointCloud_d<double, 3>, 3>
    kd_tree_3;
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 7>>,
    nanoflann::PointCloud_d<double, 7>, 7>
    kd_tree_7;

#define doubleMAX (1.79769e308)
#define doubleConserMAX (1.79760e308)

struct mutex_wrapper : std::mutex
{
    mutex_wrapper() = default;
    mutex_wrapper(mutex_wrapper const &) noexcept : std::mutex() {}
    bool operator==(mutex_wrapper const &other) noexcept { return this == &other; }
};

struct condition_variable_wrapper : std::condition_variable
{
    condition_variable_wrapper() = default;
    condition_variable_wrapper(condition_variable_wrapper const &) noexcept : std::condition_variable() {}
};

struct status
{
    double q[7];

    bool operator==(const status &other) const
    {
        for (int32_t i = 0; i < 7; i++)
        {
            if (std::abs(q[i] - other.q[i]) >= 1e-10)
                return false;
        }
        return true;
    }
    bool operator!=(const status &other) const
    {
        for (int32_t i = 0; i < 7; i++)
        {
            if (std::abs(q[i] - other.q[i]) >= 1e-10)
                return true;
        }
        return false;
    }
    bool operator<(const status &other) const
    {
        for (int32_t i = 0; i < 6; i++)
        {
            if (q[i] != other.q[i])
                return q[i] < other.q[i];
        }
        return q[6] < other.q[6];
    }
};

// status哈希比较函数
struct status_hashCompare
{
    static std::size_t hash(const status &s)
    {
        std::size_t seed = 0;
        for (int32_t i = 0; i < 7; ++i)
            seed ^= std::hash<double>{}(s.q[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
    static bool equal(const status &s1, const status &s2)
    {
        return s1 == s2;
    }
};

struct Node
{
    double cost;
    int32_t tree;
    status par;
    std::set<status> subs;
};

typedef tbb::concurrent_hash_map<status, Node, status_hashCompare> status_concurrentMap;
typedef tbb::concurrent_hash_map<int32_t, std::pair<double, status>> treelink_concurrentMap;
typedef std::vector<treelink_concurrentMap> treegraph_concurrentMap;

struct Edge
{
    status S;
    status E;
    Edge(const status &s1, const status &s2)
    {
        if (s1 < s2)
        {
            S = s1;
            E = s2;
        }
        else
        {
            S = s2;
            E = s1;
        }
    }
    bool operator<(const Edge &other) const
    {
        if (S != other.S)
            return S < other.S;
        return E < other.E;
    }
};

// edge哈希比较函数
struct edge_hashCompare
{
    static std::size_t hash(const Edge &e)
    {
        std::size_t hS = status_hashCompare::hash(e.S);
        std::size_t hE = status_hashCompare::hash(e.E);
        return hS ^ (hE << 1); // 组合哈希值
    }
    static bool equal(const Edge &e1, const Edge &e2)
    {
        return e1.S == e2.S && e1.E == e2.E;
    }
};

typedef tbb::concurrent_hash_map<Edge, double, edge_hashCompare> edgeCost_concurrentMap;

struct edgeQe
{
    double conservatism;
    int32_t treeIndex;
    status S;
    status E;
    bool operator<(const edgeQe &other) const
    {
        return conservatism < other.conservatism;
    }
    bool operator>(const edgeQe &other) const
    {
        return conservatism > other.conservatism;
    }
};

class ConcurrentSyncCostMap
{
private:
    std::mutex mtx;
    std::mutex mtxOptCost;
    std::unordered_map<int64_t, double> data;
    int32_t minID = -1;
    double minSyncCost = 0;
    double BaseSyncRadius = 0;
    double OptCost = -1;

public:
    std::condition_variable cv;
    void setBaseSyncRadius(double r)
    {
        BaseSyncRadius = r;
    }
    // 添加或替换ID对应的值
    void setsyncSelfCost(int64_t ID, double Cost)
    {
        Cost += BaseSyncRadius;
        std::lock_guard<std::mutex> lock(mtx);
        data[ID] = Cost;
        if (Cost < minSyncCost || minID < 0)
        {
            minID = ID;
            minSyncCost = Cost;
        }
        else if (minID == ID)
        {
            int32_t IDTemp = -1;
            double CostTemp = std::numeric_limits<double>::max();
            for (const auto &pair : data)
            {
                if (pair.second < CostTemp)
                {
                    CostTemp = pair.second;
                    IDTemp = pair.first;
                }
            }
            minSyncCost = CostTemp;
            minID = IDTemp;
        }
        cv.notify_all(); // Notify all waiting threads
    }
    void setOptCost(double Cost)
    {
        std::lock_guard<std::mutex> lock(mtxOptCost);
        OptCost = Cost;
    }

    // 返回SyncCost最小的键值对
    double getsyncCost()
    {
        return minSyncCost;
    }
    // 返回OptCost最小的键值对
    double getOptCost()
    {
        return OptCost;
    }
};

struct treeInitStruct
{
    status_concurrentMap *NodeMapPointer;
    treegraph_concurrentMap *treeLinkMapPointer;
    ConcurrentSyncCostMap *minSyncCostMapPointer;

    std::function<double(const status &, const status &)> RealCostFun;
    std::function<double(const status &, const status &)> EstimateCostFun;
    std::function<double(const status &, const status &)> EdgeEstimateCostFun;
    std::function<void(const status &, std::list<status> &, const double)> Neighbor;
};

class subtreePMTG
{
private:
    double NeighborR; // 近邻查找半径
    double syncR = 0;
    bool runkey = true;

    // 需初始化的参数
    // MasterQe *masterQePointer;
    ConcurrentSyncCostMap *minSyncCostMapPointer;
    status_concurrentMap *NodeMapPointer;
    treegraph_concurrentMap *treeLinkMapPointer; // 树间的连接

    std::function<double(const status &, const status &)> RealCostFun;
    std::function<double(const status &, const status &)> EstimateCostFun;
    std::function<double(const status &, const status &)> EdgeEstimateCostFun;
    std::function<void(const status &x, std::list<status> &Xnear, const double R)> Neighbor;

public:
    status treeRoot;   // 根节点
    int32_t treeIndex; // 树的Id
    std::priority_queue<edgeQe, std::vector<edgeQe>, std::greater<edgeQe>> Qe;
    // 默认构造函数
    subtreePMTG()
    {
        // std::cout << "Default constructor called" << std::endl;
    }

    // 移动构造函数
    subtreePMTG(subtreePMTG &&other) noexcept
        : NeighborR(other.NeighborR), syncR(other.syncR),
          minSyncCostMapPointer(other.minSyncCostMapPointer), NodeMapPointer(other.NodeMapPointer),
          treeLinkMapPointer(other.treeLinkMapPointer), RealCostFun(std::move(other.RealCostFun)),
          EstimateCostFun(std::move(other.EstimateCostFun)), EdgeEstimateCostFun(std::move(other.EdgeEstimateCostFun)),
          Neighbor(std::move(other.Neighbor)), treeRoot(std::move(other.treeRoot)), treeIndex(other.treeIndex),
          Qe(std::move(other.Qe)), threadObj(std::move(other.threadObj))
    {
        // std::cout << "Move constructor called" << std::endl;
        other.minSyncCostMapPointer = nullptr;
        other.NodeMapPointer = nullptr;
        other.treeLinkMapPointer = nullptr;
    }

    // 移动赋值操作符
    subtreePMTG &operator=(subtreePMTG &&other) noexcept
    {
        if (this != &other)
        {
            NeighborR = other.NeighborR;
            syncR = other.syncR;
            minSyncCostMapPointer = other.minSyncCostMapPointer;
            NodeMapPointer = other.NodeMapPointer;
            treeLinkMapPointer = other.treeLinkMapPointer;
            RealCostFun = std::move(other.RealCostFun);
            EstimateCostFun = std::move(other.EstimateCostFun);
            EdgeEstimateCostFun = std::move(other.EdgeEstimateCostFun);
            Neighbor = std::move(other.Neighbor);
            treeRoot = std::move(other.treeRoot);
            treeIndex = other.treeIndex;
            Qe = std::move(other.Qe);
            threadObj = std::move(other.threadObj);

            other.minSyncCostMapPointer = nullptr;
            other.NodeMapPointer = nullptr;
            other.treeLinkMapPointer = nullptr;
        }
        return *this;
    }

    // 析构函数
    ~subtreePMTG()
    {
        if (threadObj.joinable())
        {
            threadObj.join();
        }
    }

    void setNeighborR(double R)
    {
        NeighborR = R;
    }

    int64_t utime_ns(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<int64_t>(1000000000UL) * static_cast<int64_t>(ts.tv_sec) +
               static_cast<int64_t>(ts.tv_nsec);
    }
    void Task() // 不再考虑子树间的自动均衡
    {
        printf("subTree%4d start work!!\r\n", treeIndex);

        ConcurrentSyncCostMap &minSyncCostMap = *minSyncCostMapPointer;
        status_concurrentMap &NodeMap = *NodeMapPointer;
        treegraph_concurrentMap &treeLinkMap = *treeLinkMapPointer;
        treelink_concurrentMap &selfTreeLink = treeLinkMap[treeIndex];
        int64_t sumtime = 0;
        std::mutex mtx;
        std::unique_lock<std::mutex> lock(mtx);
        while (Qe.size() && Qe.top().conservatism <= -1)
            Qe.pop();
        for (;;)
        {
            if (Qe.size() == 0)
            {
                minSyncCostMap.setsyncSelfCost(treeIndex, doubleMAX);
                break;
            }

            edgeQe edge = Qe.top();
            Qe.pop();
            minSyncCostMap.getsyncCost();
            int64_t tt = utime_ns();
            double edgeCost = edge.conservatism;
            minSyncCostMap.cv.wait(lock, [&minSyncCostMap, edgeCost]
                                   { return minSyncCostMap.getsyncCost() >= edgeCost; });
            sumtime += (utime_ns() - tt);
            if (edge.conservatism <= -1)
                break;
            if (edge.S == edge.E)
                break;

            double nodeS_cost;
            {
                status_concurrentMap::accessor accS;
                if (!NodeMap.find(accS, edge.S))
                    break;
                nodeS_cost = accS->second.cost;
            }

            status_concurrentMap::accessor accE;
            if (!NodeMap.find(accE, edge.E))
                break;

            Node &nodeE = accE->second;
            if (nodeE.tree != treeIndex && nodeE.tree != -1)
                continue;

            if (minSyncCostMap.getOptCost() > 0 &&
                nodeS_cost + EdgeEstimateCostFun(edge.S, edge.E) > minSyncCostMap.getOptCost())
                continue;
            if (nodeS_cost + EstimateCostFun(edge.S, edge.E) < nodeE.cost)
            {
                double newCost = nodeS_cost + RealCostFun(edge.S, edge.E);

                if (newCost < nodeE.cost && newCost < doubleConserMAX)
                {

                    if (nodeE.tree >= 0 && nodeE.par != edge.S)
                    {
                        status_concurrentMap::accessor accEpar;
                        NodeMap.find(accEpar, nodeE.par);
                        accEpar->second.subs.erase(edge.E);
                    }
                    {
                        status_concurrentMap::accessor accS;
                        NodeMap.find(accS, edge.S);
                        accS->second.subs.insert(edge.E);
                    }
                    nodeE.par = edge.S;
                    nodeE.cost = newCost;
                    nodeE.tree = treeIndex;
                    double nodeE_cost;
                    nodeE_cost = newCost;
                    accE.release();

                    // setsyncR(newCost);
                    minSyncCostMap.setsyncSelfCost(treeIndex, newCost);

                    std::list<status> Xnear;
                    Neighbor(edge.E, Xnear, NeighborR);
                    for (const status &x_near : Xnear) // 添加相邻节点
                    {
                        if (x_near == edge.E || x_near == edge.S)
                            continue;
                        // printf("tree %d:x_near:%lf, %lf\r\n", treeIndex, x_near.x, x_near.y);
                        status_concurrentMap::accessor accNear;
                        if (!NodeMap.find(accNear, x_near))
                            printf("NodeMap.find(accNear, x_near)!!!\r\n");
                        Node &xnear_node = accNear->second;
                        if (xnear_node.tree == -1 || xnear_node.tree == treeIndex) // 有潜力的子节点
                        {
                            double cost = nodeE_cost + EstimateCostFun(x_near, edge.E);
                            if (cost < xnear_node.cost && cost < doubleConserMAX)
                                Qe.push(edgeQe{nodeE_cost + EdgeEstimateCostFun(edge.E, x_near), treeIndex, edge.E, x_near});
                        }
                        else // 尝试优化子树间的连接
                        {
                            treelink_concurrentMap::const_accessor c_accXnear;
                            if (selfTreeLink.find(c_accXnear, xnear_node.tree)) // 已有连接存在
                            {
                                double linkCostOld = c_accXnear->second.first;
                                c_accXnear.release();

                                double linkCost = nodeE_cost + EstimateCostFun(x_near, edge.E) + xnear_node.cost;
                                if (linkCost < linkCostOld)
                                {
                                    linkCost = nodeE_cost + RealCostFun(x_near, edge.E) + xnear_node.cost;
                                    if (linkCost < linkCostOld && linkCost < doubleConserMAX)
                                    {
                                        treelink_concurrentMap::accessor accXnearInSelfTree, accSelfInXnearTree;
                                        selfTreeLink.find(accXnearInSelfTree, xnear_node.tree);
                                        treeLinkMap[xnear_node.tree].find(accSelfInXnearTree, treeIndex);
                                        accXnearInSelfTree->second = {linkCost, edge.E}; // 添加对邻树的连接
                                        accSelfInXnearTree->second = {linkCost, x_near}; // 添加对本树的连接
                                        if (minSyncCostMap.getOptCost() > 0 && linkCost < minSyncCostMap.getOptCost())
                                            minSyncCostMap.setOptCost(linkCost);
                                    }
                                }
                            }
                            else // 首次连接
                            {
                                double linkCost = nodeE_cost + RealCostFun(x_near, edge.E) + xnear_node.cost;
                                if (linkCost < doubleConserMAX)
                                {
                                    selfTreeLink.insert({xnear_node.tree, {linkCost, edge.E}});
                                    treeLinkMap[xnear_node.tree].insert({treeIndex, {linkCost, x_near}});
                                    if (minSyncCostMap.getOptCost() > 0 && linkCost < minSyncCostMap.getOptCost())
                                        minSyncCostMap.setOptCost(linkCost);
                                }
                            }
                        }
                    }
                }
            }
            // sumtime += (utime_ns() - t0);
        }
        printf("subTree%4d, %10ld stop!!\r\n", treeIndex, sumtime);
    }

    void startTask()
    {
        if (threadObj.joinable())
        {
            threadObj.join();
        }
        threadObj = std::thread(&subtreePMTG::Task, this);
    }

    void AddInitPotentialNode()
    {
        status_concurrentMap &NodeMap = *NodeMapPointer;
        std::list<status> Xnear;
        Neighbor(treeRoot, Xnear, NeighborR);
        for (const status &x_near : Xnear) // 添加相邻节点
        {
            if (x_near == treeRoot)
                continue;
            status_concurrentMap::accessor accNear;
            NodeMap.find(accNear, x_near);
            Node &xnear_node = accNear->second;
            if (xnear_node.tree == -1 || xnear_node.tree == treeIndex) // 有潜力的子节点
            {
                double cost = EdgeEstimateCostFun(treeRoot, x_near);
                if (cost < xnear_node.cost && cost < doubleConserMAX)
                    Qe.push({cost, treeIndex, treeRoot, x_near});
            }
        }
    }

    void InitTree(int32_t id, status root, treeInitStruct &initStruct)
    {
        treeIndex = id;
        treeRoot = root;
        // masterQePointer = initStruct.masterQePointer;
        NodeMapPointer = initStruct.NodeMapPointer;
        treeLinkMapPointer = initStruct.treeLinkMapPointer;
        RealCostFun = initStruct.RealCostFun;
        EstimateCostFun = initStruct.EstimateCostFun;
        EdgeEstimateCostFun = initStruct.EdgeEstimateCostFun;
        Neighbor = initStruct.Neighbor;
        minSyncCostMapPointer = initStruct.minSyncCostMapPointer;
    }

    void stopTask()
    {
        minSyncCostMapPointer->setsyncSelfCost(treeIndex, doubleMAX);
        Qe.push({-10, -1, {}, {}});
    }
    void joinTask()
    {
        if (threadObj.joinable())
            threadObj.join(); // 等待子线程结束
    }
    void stopjoinTask()
    {
        stopTask();
        joinTask();
    }

    std::thread threadObj; // 线程对象
};

struct BoxSpace
{
    double x, y, z;
    double lenx, leny, lenz;
};
struct SphereSpace
{
    double x, y, z, r;
};
struct CylinderSpace
{
    double x, y, z, r, h;
};

#define statusDimension 7

struct statusSpaceStruct
{
    double spatialExtentMin[statusDimension];
    double spatialExtentMax[statusDimension];

    nanoflann::PointCloud_d<double, 3> ObsTreeCloud;
};

class PMTG7DOF
{

private:
    int32_t sampleN = 10000;
    nanoflann::PointCloud_d<double, 3> ObsTreeCloud;
    nanoflann::PointCloud_d<double, 7> SearchTreeCloud;
    kd_tree_3 obsSearchTree;
    kd_tree_7 NeighborSearchTree;
    double spatialExtentMin[statusDimension];
    double spatialExtentMax[statusDimension];
    double frontierMin, frontierMax;
    ConcurrentSyncCostMap minSyncCostMap;

public:
    // std::vector<status> objectPointSet;
    // cv::Mat Freemap; // free图象显示
    // cv::Mat Obsmap;  // obs图象碰撞检测
    statusSpaceStruct statusSpace;
    uGen3Ctrl Gen3obj;
    bool oc = false;

    std::vector<subtreePMTG> treelist;    // 子树列表
    status_concurrentMap NodeMap;        // 高并发状态字典
    treegraph_concurrentMap treeLinkMap; // 树间的连接
    edgeCost_concurrentMap edgeCostMap;  // 边代价字典
    status_concurrentMap RGGOptNodeMap;
    std::vector<subtreePMTG> RGGOptSubTreeList;
    treegraph_concurrentMap RGGOpttreeLinkMap;
    PMTG7DOF() : obsSearchTree(3, ObsTreeCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)),
                NeighborSearchTree(7, SearchTreeCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10))
    {
        RGGOptSubTreeList.resize(2);
        RGGOpttreeLinkMap.resize(2);
    }
    long long utime_ns(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return static_cast<long long>(1000000000UL) * static_cast<long long>(ts.tv_sec) +
               static_cast<long long>(ts.tv_nsec);
    }
    void InitRootSampleStatus(std::vector<status> &Xroot, int32_t rootNum, double R) //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    {
        R = R * R;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribI(0, 1);
        Node sampleNode = {0, -1, {}, {}};
        int32_t newStatusNum = 0;
        int32_t sampleNumMax = 3 * sampleN;
        while (sampleNumMax-- > 0)
        {
            status x_sam;
            nanoflann::PointCloud_d<double, 7>::Point query_pt;
            for (int32_t i = 0; i < statusDimension; i++)
            {
                x_sam.q[i] = (spatialExtentMax[i] - spatialExtentMin[i]) * distribI(gen) + spatialExtentMin[i];
                query_pt.q[i] = x_sam.q[i];
            }
            if (!ValidityCheck(x_sam))
                continue;

            std::vector<size_t> ret_indexes(1);
            std::vector<double> out_dists_sqr(1);
            nanoflann::KNNResultSet<double> resultSet(1);
            resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
            NeighborSearchTree.findNeighbors(resultSet, &query_pt.q[0]);

            if (resultSet.size() == 0 || out_dists_sqr[0] > R)
            {
                SearchTreeCloud.pts.push_back(query_pt);
                NeighborSearchTree.addPoints(SearchTreeCloud.pts.size() - 1, SearchTreeCloud.pts.size() - 1);
                Xroot.push_back(x_sam);
                newStatusNum++;
            }

            if (newStatusNum >= rootNum)
                break;
        }
    }

    void BatchSampleStatus()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribI(0, 1);
        Node sampleNode = {doubleMAX, -1, {}, {}};

        uint32_t startIndex = SearchTreeCloud.pts.size();
        int32_t newStatusNum = 0;
        int32_t sampleNumMax = 3 * sampleN;

        while (sampleNumMax-- > 0)
        {
            status x_sam;
            nanoflann::PointCloud_d<double, 7>::Point query_pt;
            for (int32_t i = 0; i < statusDimension; i++)
            {
                x_sam.q[i] = (spatialExtentMax[i] - spatialExtentMin[i]) * distribI(gen) + spatialExtentMin[i];
                query_pt.q[i] = x_sam.q[i];
            }
            if (!ValidityCheck(x_sam))
                continue;
            newStatusNum++;

            SearchTreeCloud.pts.push_back(query_pt);

            NodeMap.insert({x_sam, sampleNode}); //!!!!!!!!!!!!!!!!!!!!!!!

            if (newStatusNum >= sampleN)
                break;
        }
        NeighborSearchTree.addPoints(startIndex, SearchTreeCloud.pts.size() - 1);
    }

    bool ValidityCheck(const status &x)
    {
        double query_pt[3];
        std::vector<Eigen::Matrix4d> Gen3_TMat(9);
        VectorXd Q(7);
        Q << x.q[0], x.q[1], x.q[2], x.q[3], x.q[4], x.q[5], x.q[6];

        MatrixXd outmat = Gen3obj.Gen3_B_T_1(Q[0]);
        std::vector<MatrixXd> mat_1_H = {Gen3obj.Gen3_1_T_2(Q[1]), Gen3obj.Gen3_2_T_3(Q[2]),
                                         Gen3obj.Gen3_3_T_4(Q[3]), Gen3obj.Gen3_4_T_5(Q[4]),
                                         Gen3obj.Gen3_5_T_6(Q[5]), Gen3obj.Gen3_6_T_7(Q[6]),
                                         Gen3obj.Gen3_7_T_H(), Gen3obj.Gen3_H_T_D()};
        for (int i = 0; i < 8; i++)
        {
            Gen3_TMat[i] = outmat;
            outmat = outmat * mat_1_H[i];
        }
        Gen3_TMat[8] = outmat;

        if (ObsTreeCloud.pts.size())
        {
            std::vector<size_t> ret_indexes(1);
            std::vector<double> out_dists_sqr(1);
            nanoflann::KNNResultSet<double> resultSet(1);
            for (int32_t i = 1; i < Gen3_TMat.size(); i++)
            {
                Vector3d p0, p1;
                auto &TM0 = Gen3_TMat[i - 1];
                p0 = TM0.block<3, 1>(0, 3);

                auto &TM1 = Gen3_TMat[i];
                p1 = TM1.block<3, 1>(0, 3);
                int32_t n = (p0 - p1).norm() / 0.09 + 1;
                for (int32_t j = 0; j < n; j++)
                {
                    double t = (double)j / n;
                    Vector3d p = t * p1 + (1 - t) * p0;
                    query_pt[0] = p[0];
                    query_pt[1] = p[1];
                    query_pt[2] = p[2];
                    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
                    obsSearchTree.findNeighbors(resultSet, &query_pt[0]);
                    if (out_dists_sqr[0] < 0.07 * 0.07)
                        return false;
                }
            }
        }

        Vector3d p0, p1, p2, p3, p4;
        p0 = {0, 0, 0};
        p1 = Gen3_TMat[1].block<3, 1>(0, 3);
        p2 = Gen3_TMat[3].block<3, 1>(0, 3);
        p3 = Gen3_TMat[5].block<3, 1>(0, 3);
        p4 = Gen3_TMat[8].block<3, 1>(0, 3);
        if (distanceBetweenSegments(p0, p1, p3, p4) < 0.12)
            return false;
        if (distanceBetweenSegments(p2, p1, p3, p4) < 0.12)
            return false;
        return true;
    }

    double RealCostFun(const status &S, const status &E)
    {
        edgeCost_concurrentMap::const_accessor c_accCost;
        if (edgeCostMap.find(c_accCost, {S, E}) && oc == false)
        {
            return c_accCost->second;
        }
        else
        {

            double realCost = 0;
            for (int32_t i = 0; i < statusDimension; i++)
            {
                double dq = S.q[i] - E.q[i];
                realCost += dq * dq;
            }
            realCost = sqrt(realCost);

            if (!(ValidityCheck(S) && ValidityCheck(E)))
            {
                edgeCostMap.insert({{S, E}, doubleMAX});
                return doubleMAX;
            }

            bool key = false;
            double len = realCost;
            int32_t MaximumDiscrete = len / 0.09 + 2; //$$$

            for (int32_t i = 1; i < MaximumDiscrete; i++)
            {
                double t = (double)i / MaximumDiscrete;
                status xtemp;
                for (int32_t q_i = 0; q_i < statusDimension; q_i++)
                    xtemp.q[q_i] = E.q[q_i] * t + (1 - t) * S.q[q_i];
                if (!ValidityCheck(xtemp))
                {
                    edgeCostMap.insert({{S, E}, doubleMAX});
                    return doubleMAX;
                }
            }
            // for (double denominator = 2; denominator <= MaximumDiscrete; denominator *= 2)
            // {
            //     for (int32_t i = 1; i < denominator; i += 2)
            //     {
            //         double t = i / denominator;
            //         status xtemp;
            //         for (int32_t q_i = 0; q_i < statusDimension; q_i++)
            //             xtemp.q[q_i] = S.q[q_i] * t + (1 - t) * E.q[q_i];

            //         if (!ValidityCheck(xtemp))
            //         {
            //             edgeCostMap.insert({{S, E}, doubleMAX});
            //             return doubleMAX;
            //         }
            //     }
            // }

            edgeCostMap.insert({{S, E}, realCost});
            return realCost;
        }
    }
    double originalEstimateCostFun(const status &S, const status &E)
    {
        double realCost = 0;
        for (int32_t i = 0; i < statusDimension; i++)
        {
            double dq = S.q[i] - E.q[i];
            realCost += dq * dq;
        }
        return sqrt(realCost);
    }
    double EstimateCostFun(const status &S, const status &E)
    {
        edgeCost_concurrentMap::const_accessor c_accCost;
        if (edgeCostMap.find(c_accCost, {S, E}))
        {
            return c_accCost->second;
        }
        else
        {
            double realCost = 0;
            for (int32_t i = 0; i < statusDimension; i++)
            {
                double dq = S.q[i] - E.q[i];
                realCost += dq * dq;
            }
            return sqrt(realCost);
        }
    }
    void Neighbor(const status &x, std::list<status> &Xnear, const double R)
    {
        Xnear.clear();

        nanoflann::PointCloud_d<double, 7>::Point query_pt;
        for (int32_t i = 0; i < statusDimension; i++)
            query_pt.q[i] = x.q[i];
        std::vector<nanoflann::ResultItem<uint32_t, double>> IndicesDists;
        nanoflann::RadiusResultSet<double, uint32_t> resultSet(R, IndicesDists);

        NeighborSearchTree.findNeighbors(resultSet, query_pt.q);
        for (const auto &result : IndicesDists)
        {
            double *q_temp = SearchTreeCloud.pts[result.first].q;

            Xnear.emplace_back();
            double *x_temp = Xnear.back().q;
            for (int32_t i = 0; i < statusDimension; i++)
                x_temp[i] = q_temp[i];
        }
    }

    void InitializeInvironment(statusSpaceStruct &space)
    {
        statusSpace = space;
        frontierMin = doubleMAX;
        frontierMax = -doubleMAX;
        for (int32_t i = 0; i < statusDimension; i++)
        {
            spatialExtentMin[i] = statusSpace.spatialExtentMin[i];
            spatialExtentMax[i] = statusSpace.spatialExtentMax[i];
            double spatialExtent = abs(spatialExtentMax[i] - spatialExtentMin[i]);
            if (spatialExtent < frontierMin)
                frontierMin = spatialExtent;
            if (spatialExtent > frontierMax)
                frontierMax = spatialExtent;
        }
        if (space.ObsTreeCloud.pts.size())
        {
            ObsTreeCloud = space.ObsTreeCloud;
            obsSearchTree.addPoints(0, ObsTreeCloud.pts.size() - 1);
        }
        printf("frontierMin: %f\r\n", frontierMin);
        printf("frontierMax: %f\r\n", frontierMax);
    }
    double unitNBallMeasure(unsigned int N)
    {
        return std::pow(std::sqrt(M_PI), static_cast<double>(N)) /
               std::tgamma(static_cast<double>(N) / 2.0 + 1.0);
    }
    double RootNodeRadius(int32_t rootNum)
    {
        return std::pow(0.3 * std::pow(frontierMin, statusDimension) / rootNum, 1.0 / statusDimension);
    }

    double ConnectionRadius()
    {
        double q = NodeMap.size();
        double _1_n = 1.0 / statusDimension;
        double radius_1 = _1_n;
        double radius_2 = std::pow(frontierMax, statusDimension) * unitNBallMeasure(statusDimension);
        double radius_3 = log(q) / q;
        std::cout << "ConnectionRadius: " << 1.2 * pow(2 * radius_1 * radius_2 * radius_3, _1_n) << std::endl;
        return 1.5 * pow(2 * radius_1 * radius_2 * radius_3, _1_n);
    }

    double getsyncCostRadius()
    {
        return 1.1 * std::min(RootNodeRadius(treelist.size()), ConnectionRadius());
    }

    void InitializeSubTree(int32_t subTreeNum, int32_t BsamNum = 10000)
    {
        treeInitStruct initSubTreeStruct;
        initSubTreeStruct.EstimateCostFun = std::bind(&PMTG7DOF::EstimateCostFun, this,
                                                      std::placeholders::_1,
                                                      std::placeholders::_2);
        initSubTreeStruct.EdgeEstimateCostFun = std::bind(&PMTG7DOF::EstimateCostFun, this,
                                                          std::placeholders::_1,
                                                          std::placeholders::_2);
        initSubTreeStruct.RealCostFun = std::bind(&PMTG7DOF::RealCostFun, this,
                                                  std::placeholders::_1,
                                                  std::placeholders::_2);
        initSubTreeStruct.Neighbor = std::bind(&PMTG7DOF::Neighbor, this,
                                               std::placeholders::_1,
                                               std::placeholders::_2,
                                               std::placeholders::_3);
        initSubTreeStruct.minSyncCostMapPointer = &minSyncCostMap;
        initSubTreeStruct.NodeMapPointer = &NodeMap;
        initSubTreeStruct.treeLinkMapPointer = &treeLinkMap;

        std::vector<status> Xroot;
        double rootSaftR = RootNodeRadius(subTreeNum);
        InitRootSampleStatus(Xroot, subTreeNum, rootSaftR);
        sampleN = BsamNum;
        // BatchSampleStatus();

        // minSyncCostMap.setBaseSyncRadius(getsyncCostRadius());

        treelist.resize(Xroot.size());
        treeLinkMap.resize(Xroot.size());
        // double nearR = ConnectionRadius();
        for (int32_t subTreeIndex = 0; subTreeIndex < Xroot.size(); subTreeIndex++)
        {
            minSyncCostMap.setsyncSelfCost(subTreeIndex, 0);
            status &root = Xroot[subTreeIndex];
            NodeMap.insert({root, {0, subTreeIndex, {-1, -1, -1}, {}}});
            subtreePMTG &subTree = treelist[subTreeIndex];
            subTree.InitTree(subTreeIndex, root, initSubTreeStruct);
        }
        printf("Xroot.size(): %ld\r\n", Xroot.size());
    }

    void SupplementaryRun()
    {
        BatchSampleStatus();
        startSubTask();
    }

    void startSubTask()
    {
        double nearR = ConnectionRadius();
        minSyncCostMap.setBaseSyncRadius(getsyncCostRadius());
        for (subtreePMTG &subtree : treelist)
        {
            minSyncCostMap.setsyncSelfCost(subtree.treeIndex, 0);
            subtree.setNeighborR(nearR);
            subtree.AddInitPotentialNode();
            subtree.startTask();
        }
    }

    void MasterTaskWait(double ms)
    {
        long long t0 = utime_ns();
        long long t1 = utime_ns();
        bool breakKey = true;
        int64_t waitTime = ms * 1e6;
        std::cout << "waitTime: " << waitTime << std::endl;
        while (utime_ns() - t0 < waitTime)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(600));
            if (utime_ns() - t1 > 5e5)
            {
                t1 = utime_ns();
                if (minSyncCostMap.getsyncCost() > doubleConserMAX)
                {
                    std::cout << "minSyncCostMap.getsyncCost:" << minSyncCostMap.getsyncCost() << std::endl;
                    breakKey = false;
                    break;
                }
            }
        }
        if (breakKey)
            stopSubTask();
        printf("kkk,%10ld\r\n", (int64_t)(utime_ns() - t0));
    }

    double GetSubTreeGraphPath(const status S, const status E, std::list<status> &Path) // A_Star
    {
        Path.clear();
        double nearR = ConnectionRadius();
        double minNeaarCost;
        std::list<status> Xnear;

        std::priority_queue<std::pair<double, int32_t>, // fCost, treeIndex
                            std::vector<std::pair<double, int32_t>>,
                            std::greater<std::pair<double, int32_t>>>
            openSetQue;
        std::map<int32_t, std::pair<double, int32_t>> openSetSet; // treeIndex gCost, parTreeIndex
        std::map<int32_t, std::pair<double, status>> NearTreeS, NearTreeE;
        std::map<int32_t, std::pair<double, int32_t>> closedSetAStar; // treeIndex gCost, parTreeIndex

        minNeaarCost = doubleMAX;
        Neighbor(S, Xnear, nearR);
        for (const status &x : Xnear)
        {
            status_concurrentMap::accessor accNear;
            if (NodeMap.find(accNear, x))
            {
                double costTemp = accNear->second.cost + RealCostFun(x, S); // EstimateCostFun(x, S);//
                int32_t treeIndex = accNear->second.tree;
                if (costTemp < doubleConserMAX)
                {
                    auto it = NearTreeS.find(treeIndex);
                    if (it == NearTreeS.end() || it->second.first > costTemp)
                        NearTreeS[treeIndex] = {costTemp, x};
                    break;
                }
            }
        }
        minNeaarCost = doubleMAX;
        Neighbor(E, Xnear, nearR);
        for (const status &x : Xnear)
        {
            status_concurrentMap::accessor accNear;
            if (NodeMap.find(accNear, x))
            {
                double costTemp = accNear->second.cost + RealCostFun(x, E); // EstimateCostFun(x, S);//
                int32_t treeIndex = accNear->second.tree;
                if (costTemp < doubleConserMAX)
                {
                    auto it = NearTreeE.find(treeIndex);
                    if (it == NearTreeE.end() || it->second.first > costTemp)
                        NearTreeE[treeIndex] = {costTemp, x};
                    break;
                }
            }
        }
        if (NearTreeS.size() == 0 || NearTreeS.size() == 0)
            return doubleMAX;

        // 初始化
        for (const auto &pair : NearTreeS)
        {
            int32_t treeIndex = pair.first;
            double gCost = pair.second.first;
            double linkFCost = gCost +
                               EstimateCostFun(E, treelist[treeIndex].treeRoot);
            openSetSet[treeIndex] = {gCost, -1};
            openSetQue.push({linkFCost, treeIndex});
        }
        bool HavePath = false;
        while (openSetSet.size())
        {
            int32_t current_Index = openSetQue.top().second;
            double current_fCost = openSetQue.top().first;
            openSetQue.pop();
            if (openSetSet.count(current_Index) == 0)
                continue;
            double current_gCost = openSetSet[current_Index].first;
            closedSetAStar[current_Index] = openSetSet[current_Index];
            openSetSet.erase(current_Index);
            if (current_Index == -2)
            {
                HavePath = true;
                break;
            }

            for (const auto &treeLink : treeLinkMap[current_Index])
            {
                int32_t linkTreeIndex = treeLink.first;
                if (closedSetAStar.count(linkTreeIndex))
                    continue;
                double gCost = current_gCost + treeLink.second.first;
                if (openSetSet.count(linkTreeIndex) == 0 || gCost < openSetSet[linkTreeIndex].first)
                {
                    openSetSet[linkTreeIndex] = {gCost, current_Index};
                    double link_fCost = gCost + EstimateCostFun(E, treelist[linkTreeIndex].treeRoot);
                    // if (openSetSet.count(linkTreeIndex) == 0)
                    openSetQue.push({link_fCost, linkTreeIndex});
                }
            }
            if (NearTreeE.count(current_Index))
            {
                int32_t linkTreeIndex = -2;
                if (closedSetAStar.count(linkTreeIndex))
                    continue;
                double gCost = current_gCost + NearTreeE[current_Index].first;
                if (openSetSet.count(linkTreeIndex) == 0 || gCost < openSetSet[linkTreeIndex].first)
                {
                    openSetSet[linkTreeIndex] = {gCost, current_Index};
                    double link_fCost = gCost + EstimateCostFun(E, treelist[linkTreeIndex].treeRoot);
                    // if (openSetSet.count(linkTreeIndex) == 0)
                    openSetQue.push({link_fCost, linkTreeIndex});
                }
            }
        }
        if (!HavePath)
            return doubleMAX;
        // 获取路径
        std::list<status> PathTemp;
        int32_t treeIndexTemp = closedSetAStar[-2].second;
        Path.push_front(E);
        status treeRootTemp = treelist[treeIndexTemp].treeRoot;
        status xTemp = NearTreeE[treeIndexTemp].second;
        PathTemp.push_front(xTemp);
        while (xTemp != treeRootTemp)
        {
            status_concurrentMap::const_accessor c_accNodeMap;
            NodeMap.find(c_accNodeMap, xTemp);
            xTemp = c_accNodeMap->second.par;
            PathTemp.push_front(xTemp);
        }
        Path.splice(Path.begin(), PathTemp);
        while (closedSetAStar[treeIndexTemp].second != -1)
        {
            int32_t treeIndexOld = treeIndexTemp;
            treeIndexTemp = closedSetAStar[treeIndexTemp].second;
            treelink_concurrentMap::const_accessor c_accLinkOld;
            treeLinkMap[treeIndexOld].find(c_accLinkOld, treeIndexTemp);
            treeRootTemp = treelist[treeIndexOld].treeRoot;
            xTemp = c_accLinkOld->second.second;
            while (xTemp != treeRootTemp)
            {
                PathTemp.push_back(xTemp);
                status_concurrentMap::const_accessor c_accNodeMap;
                NodeMap.find(c_accNodeMap, xTemp);
                xTemp = c_accNodeMap->second.par;
            }
            Path.splice(Path.begin(), PathTemp);

            treelink_concurrentMap::const_accessor c_accLinkTemp;
            treeLinkMap[treeIndexTemp].find(c_accLinkTemp, treeIndexOld);
            treeRootTemp = treelist[treeIndexTemp].treeRoot;
            xTemp = c_accLinkTemp->second.second;
            do
            {
                status_concurrentMap::const_accessor c_accNodeMap;
                NodeMap.find(c_accNodeMap, xTemp);
                xTemp = c_accNodeMap->second.par;
                PathTemp.push_front(xTemp);
            } while (xTemp != treeRootTemp);
            Path.splice(Path.begin(), PathTemp);
        }
        treeRootTemp = treelist[treeIndexTemp].treeRoot;
        xTemp = NearTreeS[treeIndexTemp].second;
        while (xTemp != treeRootTemp)
        {
            PathTemp.push_back(xTemp);
            status_concurrentMap::const_accessor c_accNodeMap;
            NodeMap.find(c_accNodeMap, xTemp);
            xTemp = c_accNodeMap->second.par;
        }
        Path.splice(Path.begin(), PathTemp);
        Path.push_front(S);
        return closedSetAStar[-2].first;
    }
    double GetRGGOptPath(const status S, const status E, std::list<status> &Path, double OptCost = -1) // MT*
    {
        Path.clear();
        double nearR = ConnectionRadius();
        auto EdgeEstimateCostFunS = [this, &E](const status &EdgeS, const status &EdgeE)
        { return EstimateCostFun(EdgeS, EdgeE) + originalEstimateCostFun(EdgeE, E); };
        auto EdgeEstimateCostFunE = [this, &S](const status &EdgeS, const status &EdgeE)
        { return EstimateCostFun(EdgeS, EdgeE) + originalEstimateCostFun(EdgeE, S); };

        RGGOptNodeMap.clear();
        Node sampleNode = {doubleMAX, -1, {}, {}};
        RGGOptNodeMap.insert({S, {0, 0, {}, {}}});
        RGGOptNodeMap.insert({E, {0, 1, {}, {}}});
        for (auto &node : NodeMap)
            RGGOptNodeMap.insert({node.first, sampleNode});
        ConcurrentSyncCostMap RGGOptminSyncCostMap;
        RGGOptminSyncCostMap.setBaseSyncRadius(doubleMAX / 2);
        RGGOptminSyncCostMap.setsyncSelfCost(0, 0);
        RGGOptminSyncCostMap.setsyncSelfCost(1, 0);
        RGGOptminSyncCostMap.setOptCost(OptCost);

        treeInitStruct initSubTreeStructS, initSubTreeStructE;
        initSubTreeStructS.NodeMapPointer = &RGGOptNodeMap;
        initSubTreeStructS.treeLinkMapPointer = &RGGOpttreeLinkMap;
        initSubTreeStructS.minSyncCostMapPointer = &RGGOptminSyncCostMap;
        initSubTreeStructS.Neighbor = std::bind(&PMTG7DOF::Neighbor, this,
                                                std::placeholders::_1,
                                                std::placeholders::_2,
                                                std::placeholders::_3);
        initSubTreeStructS.EdgeEstimateCostFun = EdgeEstimateCostFunS;
        initSubTreeStructS.RealCostFun = std::bind(&PMTG7DOF::RealCostFun, this,
                                                   std::placeholders::_1,
                                                   std::placeholders::_2);
        initSubTreeStructS.EstimateCostFun = std::bind(&PMTG7DOF::EstimateCostFun, this,
                                                       std::placeholders::_1,
                                                       std::placeholders::_2);
        initSubTreeStructE.NodeMapPointer = &RGGOptNodeMap;
        initSubTreeStructE.treeLinkMapPointer = &RGGOpttreeLinkMap;
        initSubTreeStructE.minSyncCostMapPointer = &RGGOptminSyncCostMap;
        initSubTreeStructE.Neighbor = std::bind(&PMTG7DOF::Neighbor, this,
                                                std::placeholders::_1,
                                                std::placeholders::_2,
                                                std::placeholders::_3);
        initSubTreeStructE.EdgeEstimateCostFun = EdgeEstimateCostFunE;
        initSubTreeStructE.RealCostFun = std::bind(&PMTG7DOF::RealCostFun, this,
                                                   std::placeholders::_1,
                                                   std::placeholders::_2);
        initSubTreeStructE.EstimateCostFun = std::bind(&PMTG7DOF::EstimateCostFun, this,
                                                       std::placeholders::_1,
                                                       std::placeholders::_2);
        RGGOptSubTreeList[0].InitTree(0, S, initSubTreeStructS);
        RGGOptSubTreeList[0].setNeighborR(nearR);
        RGGOptSubTreeList[0].AddInitPotentialNode();
        RGGOptSubTreeList[1].InitTree(1, E, initSubTreeStructE);
        RGGOptSubTreeList[1].setNeighborR(nearR);
        RGGOptSubTreeList[1].AddInitPotentialNode();
        RGGOptSubTreeList[1].startTask();
        RGGOptSubTreeList[0].startTask();

        long long t0 = utime_ns();
        long long t1 = utime_ns();
        bool breakKey = true;
        int64_t waitTime = 10 * 1e9;
        std::cout << "waitTime: " << waitTime << std::endl;
        while (utime_ns() - t0 < waitTime)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(600));
            if (utime_ns() - t1 > 5e5)
            {
                t1 = utime_ns();
                if (RGGOptminSyncCostMap.getsyncCost() > doubleConserMAX || RGGOptminSyncCostMap.getOptCost() < 4.7*1.02)
                {
                    breakKey = false;
                    break;
                }
            }
        }

        RGGOptSubTreeList[0].stopTask();
        RGGOptSubTreeList[1].stopTask();
        RGGOptSubTreeList[0].joinTask();
        RGGOptSubTreeList[1].joinTask();
        treelink_concurrentMap::const_accessor c_accLinkS, c_accLinkE;
        RGGOpttreeLinkMap[0].find(c_accLinkS, 1);
        RGGOpttreeLinkMap[1].find(c_accLinkE, 0);
        status statusTemp = c_accLinkS->second.second;
        status treeRootTemp = RGGOptSubTreeList[0].treeRoot;
        while (statusTemp != treeRootTemp)
        {
            Path.push_front(statusTemp);
            status_concurrentMap::const_accessor c_accNodeMap;
            RGGOptNodeMap.find(c_accNodeMap, statusTemp);
            statusTemp = c_accNodeMap->second.par;
        }
        Path.push_front(treeRootTemp);
        statusTemp = c_accLinkE->second.second;
        treeRootTemp = RGGOptSubTreeList[1].treeRoot;
        while (statusTemp != treeRootTemp)
        {
            Path.push_back(statusTemp);
            status_concurrentMap::const_accessor c_accNodeMap;
            RGGOptNodeMap.find(c_accNodeMap, statusTemp);
            statusTemp = c_accNodeMap->second.par;
        }
        Path.push_back(treeRootTemp);
        return c_accLinkS->second.first;
    }
    void stopSubTask()
    {
        printf("stopSubTask!! \r\n");
        for (subtreePMTG &subtree : treelist)
        {
            subtree.stopTask();
        }
        for (subtreePMTG &subtree : treelist)
            subtree.joinTask();
    }
    double distanceBetweenSegments(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector3d &D)
    {
        Vector3d u = B - A;
        Vector3d v = D - C;
        Vector3d w = A - C;

        double a = u.dot(u);
        double b = u.dot(v);
        double c = v.dot(v);
        double d = u.dot(w);
        double e = v.dot(w);

        double DD = a * c - b * b;
        double sc, sN, sD = DD;
        double tc, tN, tD = DD;

        if (DD < 1e-6)
        {
            sN = 0.0;
            sD = 1.0;
            tN = e;
            tD = c;
        }
        else
        {
            sN = (b * e - c * d);
            tN = (a * e - b * d);

            if (sN < 0.0)
            {
                sN = 0.0;
                tN = e;
                tD = c;
            }
            else if (sN > sD)
            {
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0.0)
        {
            tN = 0.0;
            if (-d < 0.0)
            {
                sN = 0.0;
            }
            else if (-d > a)
            {
                sN = sD;
            }
            else
            {
                sN = -d;
                sD = a;
            }
        }
        else if (tN > tD)
        {
            tN = tD;
            if ((-d + b) < 0.0)
            {
                sN = 0;
            }
            else if ((-d + b) > a)
            {
                sN = sD;
            }
            else
            {
                sN = (-d + b);
                sD = a;
            }
        }

        sc = (fabs(sN) < 1e-6 ? 0.0 : sN / sD);
        tc = (fabs(tN) < 1e-6 ? 0.0 : tN / tD);

        Vector3d dP = w + (sc * u) - (tc * v);

        return dP.norm();
    }

    void showPath(std::list<status> &Path)
    {
        std::list<status> PathShow;
        PathShow.push_back(Path.front());
        status p_old = Path.front();
        double discreteDistance = 0.02;
        for (const status &p_now : Path)
        {
            double realCost = 0;
            for (int32_t i = 0; i < statusDimension; i++)
            {
                double dq = p_now.q[i] - p_old.q[i];
                realCost += dq * dq;
            }
            realCost = sqrt(realCost);

            double len = realCost;

            int32_t MaximumDiscrete = len / discreteDistance + 2; //$$$
            for (int32_t i = 1; i <= MaximumDiscrete; i++)
            {

                double t = (double)i / MaximumDiscrete;
                status xtemp;
                for (int32_t i = 0; i < statusDimension; i++)
                    xtemp.q[i] = p_now.q[i] * t + (1 - t) * p_old.q[i];

                if (originalEstimateCostFun(xtemp, PathShow.back()) > 0.01)
                    PathShow.push_back(xtemp);
            }
            p_old = p_now;
        }
        PathShow.push_back(p_old);
        Path.clear();
        Path.splice(Path.end(), PathShow);
    }
};

#endif
