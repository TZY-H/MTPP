#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/util/PPM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>

#include <ompl/base/ProblemDefinition.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <Eigen/Dense>

#include "statusSpace7DOF.h"
#include "uGen3Ctrl.h"

// #define udebug
#define uptime 60.0
#define ratioCost 1.05

namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStateValid(const ob::State *state) {
    const ob::RealVectorStateSpace::StateType *rvState = state->as<ob::RealVectorStateSpace::StateType>();
    double query_pt[3];
    std::vector<Eigen::Matrix4d> Gen3_TMat(9);
    VectorXd Q(7);
    Q << rvState->values[0],rvState->values[1],
         rvState->values[2],rvState->values[3],
         rvState->values[4],rvState->values[5],
         rvState->values[6];

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
        for (size_t i = 1; i < Gen3_TMat.size(); i++)
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


void plan(const BItask &task, int64_t &ti, int64_t &to, double &ci) 
{
    auto space(std::make_shared<ob::RealVectorStateSpace>(7));
    ob::RealVectorBounds bounds(7);
    bounds.setLow( 0,  -2 * M_PI);
    bounds.setHigh(0,  2 * M_PI);
    bounds.setLow( 1,  -2 * M_PI / 3);
    bounds.setHigh(1,  2 * M_PI / 3);
    bounds.setLow( 2,  -2 * M_PI);
    bounds.setHigh(2,  2 * M_PI);
    bounds.setLow( 3,  -M_PI * 140.0 / 180.0);
    bounds.setHigh(3,  M_PI * 140.0 / 180.0);
    bounds.setLow( 4,  -M_PI * 2);
    bounds.setHigh(4,  M_PI * 2);
    bounds.setLow( 5,  -M_PI * 120.0 / 180.0);
    bounds.setHigh(5,  M_PI * 120.0 / 180.0);
    bounds.setLow( 6,  -M_PI * 2);
    bounds.setHigh(6,  M_PI * 2);

    space->setBounds(bounds);


    og::SimpleSetup ss(space); 
    ss.setStateValidityChecker([&](const ob::State *state) {
        return isStateValid(state);
    });
    ss.getSpaceInformation()->setStateValidityCheckingResolution(0.09/4/M_PI);

    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    for (size_t i = 0; i < 7; i++)
    {
        start->as<ob::RealVectorStateSpace::StateType>()->values[i] = task.S.q[i];
        goal->as<ob::RealVectorStateSpace::StateType>()->values[i] = task.G.q[i];
    }
    ss.setStartAndGoalStates(start, goal);

    auto planner(std::make_shared<og::PRMstar>(ss.getSpaceInformation()));
    ss.setPlanner(planner);

    int64_t start_time;
    double cost_tag = ratioCost * task.miniCost;

    auto problem_def = ss.getProblemDefinition();
    // 设置路径长度最优目标
    ob::OptimizationObjectivePtr objective(new ob::PathLengthOptimizationObjective(ss.getSpaceInformation()));
    objective->setCostThreshold(ompl::base::Cost(cost_tag*1000));
    problem_def->setOptimizationObjective(objective);

    ti = to = uptime * 1e9;
    start_time = utime_ns();
    ob::PlannerStatus solved = ss.solve(uptime);
    ti = utime_ns() - start_time;
    ob::Cost solution_cost = ss.getProblemDefinition()->getSolutionPath()->cost(ss.getProblemDefinition()->getOptimizationObjective());
    ci = solution_cost.value();
    ss.clear();
    objective->setCostThreshold(ompl::base::Cost(cost_tag));
    problem_def = ss.getProblemDefinition();
    problem_def->setOptimizationObjective(objective);
    start_time = utime_ns();
    solved = ss.solve(uptime);
    to = utime_ns() - start_time;

    printf("%ld, %ld ######\r\n\r\n",ti, to); 
// #ifdef udebug
    if (solved) {
        std::cout << "Found solution:" << std::endl;
        ss.simplifySolution();
        const auto &path = ss.getSolutionPath();
        path.print(std::cout);
        // ompl::base::OptimizationObjectivePtr objective;
        std::cout << "path.cost " <<path.cost(objective) <<std::endl;
        // std::stringstream namestr;
        // namestr << "/home/tzyr/WorkSpace/ompl-1.6.0/img/" << (int64_t)(utime_ns()/1e6) << ".png";
        // drawPath(map, path, namestr.str());
    } else {
        std::cout << "No solution found" << std::endl;
    }
// #endif
    (void) solved;    
}

// 打印 savedata_to
void printSavedataTo(const std::vector<int64_t> &savedata) {
        for (const auto &value : savedata) {
            std::cout << value << ", ";
        }
        std::cout << "\n";
}

// 保存 savedata_to 到文件
void saveSavedataTo(const std::map<std::string, std::vector<int64_t>> &savedata, const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto &entry : savedata) {
            file << entry.first << ":";
            for (const auto &value : entry.second) {
                file << value << ", ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << "\n";
    }
}

#define N 50
int main() {
#ifdef udebug
    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse, NULL);
#endif
    std::vector<int64_t> savedata_ti;
    std::vector<int64_t> savedata_to;
    
    BItask taskmap= {{{-1.4, 0.4, 0, 2.4, 0.2, -1.0, -1.6}}, 
                     {{0.2, 1.0, 0, 0.8, 2.0, -1.6, -2.8}}, 4.7};


    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile("./test_usrFMT7DOF/test3D/BOX2.STL",
                                             aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    double density = 16000 * 1e2;
    std::set<uSamplePoint3> point_cloud;
    MeshToPointCloud(scene, point_cloud, density);
    printf("point_cloud.size(): %ld\r\n", point_cloud.size());
    for (const auto &point : point_cloud)
        ObsTreeCloud.pts.push_back({point.x, point.y, point.z});
    obsSearchTree.addPoints(0, ObsTreeCloud.pts.size() - 1);
    std::cout << "obsSearchTree  " << ObsTreeCloud.pts.size()<<std::endl;

    int64_t data_ti=0,data_to=0;double data_Ci=0;
    for (size_t i = 0; i < N; i++)
    {
        int64_t ti, to;double ci;
        plan(taskmap, ti, to, ci);
        savedata_ti.push_back(ti);
        savedata_to.push_back(to);
        data_ti += ti;
        data_to += to;
        data_Ci += ci;
        if(i%10==0)
            printf("!!!! %ld\r\n", i);
    }
    data_ti = data_ti /N;
    data_to = data_to /N;
    data_Ci = data_Ci /N;


    std::cout << "savedata_ti: " <<std::endl; 
    printSavedataTo(savedata_ti);
    std::cout << "savedata_to: " <<std::endl; 
    printSavedataTo(savedata_to);

    std::cout << "data_ti: " << data_ti/1e6 << "  data_to: " << data_to/1e6 << "  data_Ci: " << data_Ci << std::endl;
    return 0;
}
