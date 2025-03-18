#ifndef __PMTG_H
#define __PMTG_H
#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
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
// #include <tbb/concurrent_priority_queue.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_2.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Fuzzy_sphere.h>

using Point_2 = CGAL::Exact_predicates_inexact_constructions_kernel::Point_2;
using Traits = CGAL::Search_traits_2<CGAL::Exact_predicates_inexact_constructions_kernel>;
using Kd_tree = CGAL::Kd_tree<Traits>;
using Neighbor_search = CGAL::Orthogonal_k_neighbor_search<Traits>;
using Point_with_distance = Neighbor_search::Point_with_transformed_distance;

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
    double x;
    double y;

    bool operator==(const status &other) const
    {
        return std::abs(x - other.x) < 1e-10 && std::abs(y - other.y) < 1e-10;
    }
    bool operator!=(const status &other) const
    {
        return std::abs(x - other.x) > 1e-10 || std::abs(y - other.y) > 1e-10;
    }
    bool operator<(const status &other) const
    {
        // 在这里定义比较规则
        if (x != other.x)
            return x < other.x;
        return y < other.y;
    }
};

// status哈希比较函数
struct status_hashCompare
{
    static std::size_t hash(const status &s)
    {
        std::size_t hx = std::hash<long>()((long)(s.x * 1e10));
        std::size_t hy = std::hash<long>()((long)(s.y * 1e10));
        return hx ^ (hy << 1); // 组合哈希值
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

class ConcurrentSyncCostMap2
{
private:
    std::atomic<int64_t> minID;
    std::unordered_map<int64_t, double> data;
    std::mutex dataMutex;
    double BaseSyncRadius = 0;

public:
    ConcurrentSyncCostMap2() : minID(std::numeric_limits<int64_t>::max()) {}
    std::condition_variable cv;

    void setBaseSyncRadius(double r)
    {
        BaseSyncRadius = r;
    }

    void setsyncSelfCost(int64_t ID, double Cost)
    {
        Cost += BaseSyncRadius;
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            data[ID] = Cost;
        }
        updateMinID(ID, Cost);
        cv.notify_all(); // Notify all waiting threads
    }

    double getsyncCost()
    {
        int64_t currentMinID = minID.load(std::memory_order_acquire);
        if (currentMinID == std::numeric_limits<int64_t>::max())
        {
            return std::numeric_limits<double>::infinity();
        }
        std::lock_guard<std::mutex> lock(dataMutex);
        auto it = data.find(currentMinID);
        if (it == data.end())
        {
            return std::numeric_limits<double>::infinity();
        }
        return it->second;
    }

private:
    void updateMinID(int64_t ID, double Cost)
    {
        int64_t oldMinID = minID.load(std::memory_order_relaxed);
        while (ID < oldMinID && !minID.compare_exchange_weak(oldMinID, ID, std::memory_order_acq_rel))
        {
            oldMinID = minID.load(std::memory_order_relaxed);
        }
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
            {
                printf("???\r\n");
                printf("%lf,%lf \r\n", edge.S.x, edge.S.y);
                break;
            }

            double nodeS_cost;
            {
                status_concurrentMap::accessor accS;
                if (!NodeMap.find(accS, edge.S))
                {
                    printf("%lf,%lf 1??\r\n", edge.S.x, edge.S.y);
                    break;
                }
                nodeS_cost = accS->second.cost;
            }

            status_concurrentMap::accessor accE;
            if (!NodeMap.find(accE, edge.E))
            {
                printf("%lf,%lf 2??\r\n", edge.E.x, edge.E.y);
                break;
            }

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
        Qe.push({-10, -1, {-1, -1}, {-1, -1}});
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

class PMTG2D
{
#define statusDimension 2

private:
    int32_t sampleN = 10000;
    Kd_tree NeighborSearchTree;
    double spatialExtentMin[statusDimension];
    double spatialExtentMax[statusDimension];
    double frontierMin, frontierMax;
    ConcurrentSyncCostMap minSyncCostMap;

public:
    cv::Mat Freemap; // free图象显示
    cv::Mat Obsmap;  // obs图象碰撞检测

    std::vector<subtreePMTG> treelist;    // 子树列表
    status_concurrentMap NodeMap;        // 高并发状态字典
    status_concurrentMap sNodeMap;       // 高并发状态字典
    treegraph_concurrentMap treeLinkMap; // 树间的连接
    edgeCost_concurrentMap edgeCostMap;  // 边代价字典
    status_concurrentMap RGGOptNodeMap;
    std::vector<subtreePMTG> RGGOptSubTreeList;
    treegraph_concurrentMap RGGOpttreeLinkMap;
    PMTG2D()
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
    void InitRootSampleStatus(std::vector<status> &Xroot, int32_t rootNum, double R)
    {
        R = R * R;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribX(spatialExtentMin[0], spatialExtentMax[0]);
        std::uniform_real_distribution<double> distribY(spatialExtentMin[1], spatialExtentMax[1]);
        Node sampleNode = {0, -1, {-1, -1}, {}};
        int32_t newStatusNum = 0;
        int32_t sampleNumMax = 3 * sampleN;
        while (sampleNumMax-- > 0)
        {
            status x_sam = {distribX(gen), distribY(gen)};
            if (!ValidityCheck(x_sam))
                continue;
            Neighbor_search search(NeighborSearchTree, {x_sam.x, x_sam.y}, 1);
            if (search.begin() == search.end() || search.begin()->second > R)
            {
                NeighborSearchTree.insert({x_sam.x, x_sam.y});
                NeighborSearchTree.build();
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
        std::uniform_real_distribution<double> distribX(spatialExtentMin[0], spatialExtentMax[0]);
        std::uniform_real_distribution<double> distribY(spatialExtentMin[1], spatialExtentMax[1]);
        Node sampleNode = {doubleMAX, -1, {-1, -1}, {}};
        // std::vector<status> statusSet;
        int32_t newStatusNum = 0;
        int32_t sampleNumMax = 3 * sampleN;
        while (sampleNumMax-- > 0)
        {
            status x_sam = {distribX(gen), distribY(gen)};
            if (!ValidityCheck(x_sam))
                continue;
            newStatusNum++;
            NeighborSearchTree.insert({x_sam.x, x_sam.y});
            NodeMap.insert({x_sam, sampleNode}); //!!!!!!!!!!!!!!!!!!!!!!!

            if (newStatusNum >= sampleN)
                break;
        }
        NeighborSearchTree.build();
    }

    bool ValidityCheck(const status &x)
    {
        if (Obsmap.at<uint8_t>(x.y, x.x))
            return false;
        else
            return true;
    }

    double RealCostFun(const status &S, const status &E)
    {
        edgeCost_concurrentMap::const_accessor c_accCost;
        if (edgeCostMap.find(c_accCost, {S, E}))
        {
            return c_accCost->second;
        }
        else
        {
            double dx = S.x - E.x;
            double dy = S.y - E.y;
            double realCost = sqrt(dx * dx + dy * dy);

            if (!(ValidityCheck(S) && ValidityCheck(E)))
            {
                edgeCostMap.insert({{S, E}, doubleMAX});
                return doubleMAX;
            }

            bool key = false;
            double len = realCost;
            int32_t MaximumDiscrete = len / 5 + 2; //$$$
            for (double denominator = 2; denominator <= MaximumDiscrete; denominator *= 2)
            {
                for (int32_t i = 1; i < denominator; i += 2)
                {
                    double t = i / denominator;
                    double x = S.x * t + (1 - t) * E.x;
                    double y = S.y * t + (1 - t) * E.y;
                    if (!ValidityCheck({x, y}))
                    {
                        edgeCostMap.insert({{S, E}, doubleMAX});
                        return doubleMAX;
                    }
                }
            }

            edgeCostMap.insert({{S, E}, realCost});
            return realCost;
        }
    }
    double originalEstimateCostFun(const status &S, const status &E)
    {
        double dx = S.x - E.x;
        double dy = S.y - E.y;
        return sqrt(dx * dx + dy * dy);
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
            double dx = S.x - E.x;
            double dy = S.y - E.y;
            return sqrt(dx * dx + dy * dy);
        }
    }
    void Neighbor(const status &x, std::list<status> &Xnear, const double R)
    {
        Xnear.clear();
        // 定义模糊球以进行半径搜索
        Point_2 query(x.x, x.y);
        CGAL::Fuzzy_sphere<Traits> fuzzy_sphere(query, R);
        std::vector<Point_2> result_points;
        NeighborSearchTree.search(std::back_inserter(result_points), fuzzy_sphere);
        for (const auto &point : result_points)
            Xnear.push_back({point.x(), point.y()});
    }

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
        frontierMin = min(Obsmap.cols, Obsmap.rows);
        frontierMax = max(Obsmap.cols, Obsmap.rows);
        // shapeX = Obsmap.cols;
        // shapeY = Obsmap.rows;
    }
    double unitNBallMeasure(unsigned int N)
    {
        // This is the radius version with r removed (as it is 1) for efficiency
        return std::pow(std::sqrt(boost::math::constants::pi<double>()), static_cast<double>(N)) /
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
        return 0.8 * pow(2 * radius_1 * radius_2 * radius_3, _1_n);
    }

    double getsyncCostRadius()
    {
        return 1.1 * min(RootNodeRadius(treelist.size()), ConnectionRadius());
    }

    void InitializeSubTree(int32_t subTreeNum, int32_t BsamNum = 10000)
    {
        treeInitStruct initSubTreeStruct;
        initSubTreeStruct.EstimateCostFun = std::bind(&PMTG2D::EstimateCostFun, this,
                                                      std::placeholders::_1,
                                                      std::placeholders::_2);
        initSubTreeStruct.EdgeEstimateCostFun = std::bind(&PMTG2D::EstimateCostFun, this,
                                                          std::placeholders::_1,
                                                          std::placeholders::_2);
        initSubTreeStruct.RealCostFun = std::bind(&PMTG2D::RealCostFun, this,
                                                  std::placeholders::_1,
                                                  std::placeholders::_2);
        initSubTreeStruct.Neighbor = std::bind(&PMTG2D::Neighbor, this,
                                               std::placeholders::_1,
                                               std::placeholders::_2,
                                               std::placeholders::_3);
        initSubTreeStruct.minSyncCostMapPointer = &minSyncCostMap;
        initSubTreeStruct.NodeMapPointer = &NodeMap;
        initSubTreeStruct.treeLinkMapPointer = &treeLinkMap;

        std::vector<status> Xroot;
        double rootSaftR = RootNodeRadius(subTreeNum);
        InitRootSampleStatus(Xroot, subTreeNum, rootSaftR);
        printf("Xroot.size(): %ld\r\n", Xroot.size());
        sampleN = BsamNum;
        // BatchSampleStatus();

        // minSyncCostMap.setBaseSyncRadius(getsyncCostRadius());

        treelist.resize(Xroot.size());
        treeLinkMap.resize(Xroot.size());
        double nearR = ConnectionRadius();
        for (int32_t subTreeIndex = 0; subTreeIndex < Xroot.size(); subTreeIndex++)
        {
            minSyncCostMap.setsyncSelfCost(subTreeIndex, 0);
            status &root = Xroot[subTreeIndex];
            NodeMap.insert({root, {0, subTreeIndex, {-1, -1}, {}}});
            subtreePMTG &subTree = treelist[subTreeIndex];
            subTree.InitTree(subTreeIndex, root, initSubTreeStruct);
        }
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
                double costTemp = accNear->second.cost + RealCostFun(x, S);
                int32_t treeIndex = accNear->second.tree;
                if (costTemp < doubleConserMAX)
                {
                    auto it = NearTreeS.find(treeIndex);
                    if (it == NearTreeS.end() || it->second.first > costTemp)
                        NearTreeS[treeIndex] = {costTemp, x};
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
                double costTemp = accNear->second.cost + RealCostFun(x, E);
                int32_t treeIndex = accNear->second.tree;
                if (costTemp < doubleConserMAX)
                {
                    auto it = NearTreeE.find(treeIndex);
                    if (it == NearTreeE.end() || it->second.first > costTemp)
                        NearTreeE[treeIndex] = {costTemp, x};
                }
            }
        }

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
                break;

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

        // 获取路径
        std::list<status> PathTemp;
        int32_t treeIndexTemp = closedSetAStar[-2].second;
        Path.push_front(E);
        status treeRootTemp = treelist[treeIndexTemp].treeRoot;
        status xTemp = NearTreeE[treeIndexTemp].second;
        PathTemp.push_front(xTemp);
        do
        {
            status_concurrentMap::const_accessor c_accNodeMap;
            NodeMap.find(c_accNodeMap, xTemp);
            xTemp = c_accNodeMap->second.par;
            PathTemp.push_front(xTemp);
        } while (xTemp != treeRootTemp);
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
        Node sampleNode = {doubleMAX, -1, {-1, -1}, {}};
        RGGOptNodeMap.insert({S, {0, 0, {-1, -1}, {}}});
        RGGOptNodeMap.insert({E, {0, 1, {-1, -1}, {}}});
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
        initSubTreeStructS.Neighbor = std::bind(&PMTG2D::Neighbor, this,
                                                std::placeholders::_1,
                                                std::placeholders::_2,
                                                std::placeholders::_3);
        initSubTreeStructS.EdgeEstimateCostFun = EdgeEstimateCostFunS;
        initSubTreeStructS.RealCostFun = std::bind(&PMTG2D::RealCostFun, this,
                                                   std::placeholders::_1,
                                                   std::placeholders::_2);
        initSubTreeStructS.EstimateCostFun = std::bind(&PMTG2D::EstimateCostFun, this,
                                                       std::placeholders::_1,
                                                       std::placeholders::_2);
        initSubTreeStructE.NodeMapPointer = &RGGOptNodeMap;
        initSubTreeStructE.treeLinkMapPointer = &RGGOpttreeLinkMap;
        initSubTreeStructE.minSyncCostMapPointer = &RGGOptminSyncCostMap;
        initSubTreeStructE.Neighbor = std::bind(&PMTG2D::Neighbor, this,
                                                std::placeholders::_1,
                                                std::placeholders::_2,
                                                std::placeholders::_3);
        initSubTreeStructE.EdgeEstimateCostFun = EdgeEstimateCostFunE;
        initSubTreeStructE.RealCostFun = std::bind(&PMTG2D::RealCostFun, this,
                                                   std::placeholders::_1,
                                                   std::placeholders::_2);
        initSubTreeStructE.EstimateCostFun = std::bind(&PMTG2D::EstimateCostFun, this,
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
                if (RGGOptminSyncCostMap.getsyncCost() > doubleConserMAX || RGGOptminSyncCostMap.getOptCost() < OptCost)
                {
                    breakKey = false;
                    break;
                }
            }
        }

        RGGOptSubTreeList[0].stopTask();
        RGGOptSubTreeList[1].stopTask();
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
};

#endif
