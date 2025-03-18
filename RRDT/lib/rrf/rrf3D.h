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

struct statusSpaceStruct
{
    BoxSpace boxSpace;

    std::vector<BoxSpace> obsBoxList;
    std::vector<SphereSpace> obsSphereList;
    std::vector<CylinderSpace> obsCylinderList;

    std::vector<BoxSpace> freeBoxList;
    std::vector<SphereSpace> freeSphereList;
    std::vector<CylinderSpace> freeCylinderList;
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
    // double RotationCoefficient;

    void Neighbor(const status &x, std::list<status> &Xnear, double R, kd_tree &Tree, nanoflann::PointCloud_d<double, statusDimension> &TreeCloud) //!!
    {
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

    bool ValidityCheck(const status &x)
    {
        if (x.q[0] < statusSpace.boxSpace.x ||
            x.q[1] < statusSpace.boxSpace.y ||
            x.q[2] < statusSpace.boxSpace.z ||
            x.q[0] > (statusSpace.boxSpace.x + statusSpace.boxSpace.lenx) ||
            x.q[1] > (statusSpace.boxSpace.y + statusSpace.boxSpace.leny) ||
            x.q[2] > (statusSpace.boxSpace.z + statusSpace.boxSpace.lenz))
            return false;

        for (const auto &Sphere : statusSpace.freeSphereList)
        {
            double dx = Sphere.x - x.q[0];
            double dy = Sphere.y - x.q[1];
            double dz = Sphere.z - x.q[2];
            double rr = dx * dx + dy * dy + dz * dz;
            if (rr <= Sphere.r * Sphere.r)
                return true;
        }
        for (const auto &Box : statusSpace.freeBoxList)
        {
            if (x.q[0] > Box.x && x.q[1] > Box.y && x.q[2] > Box.z &&
                (x.q[0] - Box.x) < Box.lenx &&
                (x.q[1] - Box.y) < Box.leny &&
                (x.q[2] - Box.z) < Box.lenz)
                return true;
        }
        for (const auto &Cylinder : statusSpace.freeCylinderList)
        {
            if (x.q[2] > Cylinder.z && (x.q[2] - Cylinder.z) < Cylinder.h)
            {
                double dx = x.q[0] - Cylinder.x;
                double dy = x.q[1] - Cylinder.y;
                double rr = dx * dx + dy * dy;
                if (rr <= Cylinder.r * Cylinder.r)
                    return true;
            }
        }

        for (const auto &Sphere : statusSpace.obsSphereList)
        {
            double dx = Sphere.x - x.q[0];
            double dy = Sphere.y - x.q[1];
            double dz = Sphere.z - x.q[2];
            double rr = dx * dx + dy * dy + dz * dz;
            if (rr < Sphere.r * Sphere.r)
                return false;
        }
        for (const auto &Box : statusSpace.obsBoxList)
        {
            if (x.q[0] > Box.x && x.q[1] > Box.y && x.q[2] > Box.z &&
                (x.q[0] - Box.x) < Box.lenx &&
                (x.q[1] - Box.y) < Box.leny &&
                (x.q[2] - Box.z) < Box.lenz)
                return false;
        }
        for (const auto &Cylinder : statusSpace.obsCylinderList)
        {
            if (x.q[2] > Cylinder.z && (x.q[2] - Cylinder.z) < Cylinder.h)
            {
                double dx = x.q[0] - Cylinder.x;
                double dy = x.q[1] - Cylinder.y;
                double rr = dx * dx + dy * dy;
                if (rr <= Cylinder.r * Cylinder.r)
                    return false;
            }
        }

        return true;
    }

    double HeuristicCostFun(const status &S, const status &E) //!!
    {
        double sum = 0;
        double dx = S.q[0] - E.q[0];
        double dy = S.q[1] - E.q[1];
        double dr = S.q[2] - E.q[2];
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
        int32_t MaximumDiscrete = len / 0.03 + 2; //!!
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
        query_pt.q[2] = x.q[2];
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
    statusSpaceStruct statusSpace;

    int32_t treeIndexCount = 0;

    RRFplanner(int32_t rd) : gen(rd) {}

    void InitializeInvironment(statusSpaceStruct &space)
    {
        statusSpace = space;

        spatialExtentMin[0] = statusSpace.boxSpace.x;
        spatialExtentMin[1] = statusSpace.boxSpace.y;
        spatialExtentMin[2] = statusSpace.boxSpace.z;
        spatialExtentMax[0] = statusSpace.boxSpace.x + statusSpace.boxSpace.lenx;
        spatialExtentMax[1] = statusSpace.boxSpace.y + statusSpace.boxSpace.leny;
        spatialExtentMax[2] = statusSpace.boxSpace.z + statusSpace.boxSpace.lenz;

        frontierMin = std::min(std::min(statusSpace.boxSpace.lenx, statusSpace.boxSpace.leny), statusSpace.boxSpace.lenz);
        frontierMax = std::max(std::max(statusSpace.boxSpace.lenx, statusSpace.boxSpace.leny), statusSpace.boxSpace.lenz);
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
};

long long utime_ns(void);

#endif
