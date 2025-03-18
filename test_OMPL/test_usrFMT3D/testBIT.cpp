#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/base/ProblemDefinition.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"

#include "statusSpace3D.h"

// #define udebug
#define uptime 25.0
#define ratioCost 1.02

namespace ob = ompl::base;
namespace og = ompl::geometric;
struct BIpoint
{
    double x;
    double y;
    double z;
};
struct BItask
{
    BIpoint S;
    BIpoint G;
    double miniCost;
    // std::string Path;
};

statusSpaceStruct statusSpace;

bool isStateValid(const ob::State *state) {
    const ob::RealVectorStateSpace::StateType *rvState = state->as<ob::RealVectorStateSpace::StateType>();
    BIpoint x;
    x.x = rvState->values[0];
    x.y = rvState->values[1];
    x.z = rvState->values[2];

    for (const auto &Sphere : statusSpace.freeSphereList)
    {
        double dx = Sphere.x - x.x;
        double dy = Sphere.y - x.y;
        double dz = Sphere.z - x.z;
        double rr = dx * dx + dy * dy + dz * dz;
        if (rr <= Sphere.r * Sphere.r)
            return true;
    }
    for (const auto &Box : statusSpace.freeBoxList)
    {
        if (x.x > Box.x && x.y > Box.y && x.z > Box.z &&
            (x.x - Box.x) < Box.lenx &&
            (x.y - Box.y) < Box.leny &&
            (x.z - Box.z) < Box.lenz)
            return true;
    }
    for (const auto &Cylinder : statusSpace.freeCylinderList)
    {
        if (x.z > Cylinder.z && (x.z - Cylinder.z) < Cylinder.h)
        {
            double dx = x.x - Cylinder.x;
            double dy = x.y - Cylinder.y;
            double rr = dx * dx + dy * dy;
            if (rr <= Cylinder.r * Cylinder.r)
                return true;
        }
    }

    for (const auto &Sphere : statusSpace.obsSphereList)
    {
        double dx = Sphere.x - x.x;
        double dy = Sphere.y - x.y;
        double dz = Sphere.z - x.z;
        double rr = dx * dx + dy * dy + dz * dz;
        if (rr < Sphere.r * Sphere.r)
            return false;
    }
    for (const auto &Box : statusSpace.obsBoxList)
    {
        if (x.x > Box.x && x.y > Box.y && x.z > Box.z &&
            (x.x - Box.x) < Box.lenx &&
            (x.y - Box.y) < Box.leny &&
            (x.z - Box.z) < Box.lenz)
            return false;
    }
    for (const auto &Cylinder : statusSpace.obsCylinderList)
    {
        if (x.z > Cylinder.z && (x.z - Cylinder.z) < Cylinder.h)
        {
            double dx = x.x - Cylinder.x;
            double dy = x.y - Cylinder.y;
            double rr = dx * dx + dy * dy;
            if (rr <= Cylinder.r * Cylinder.r)
                return false;
        }
    }

    return true;
}

void plan(const BItask &task, int64_t &ti, int64_t &to, double &ci) 
{
    auto space(std::make_shared<ob::RealVectorStateSpace>(3));
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0);
    bounds.setHigh(0, statusSpace.boxSpace.lenx);
    bounds.setHigh(1, statusSpace.boxSpace.leny);
    bounds.setHigh(2, statusSpace.boxSpace.lenz);
    space->setBounds(bounds);

    og::SimpleSetup ss(space); 
    ss.setStateValidityChecker([&](const ob::State *state) {
        return isStateValid(state);
    });
    ss.getSpaceInformation()->setStateValidityCheckingResolution(0.04/statusSpace.boxSpace.lenx);

    ob::ScopedState<> start(space);
    start->as<ob::RealVectorStateSpace::StateType>()->values[0] = task.S.x;
    start->as<ob::RealVectorStateSpace::StateType>()->values[1] = task.S.y;
    start->as<ob::RealVectorStateSpace::StateType>()->values[2] = task.S.z;

    ob::ScopedState<> goal(space);
    goal->as<ob::RealVectorStateSpace::StateType>()->values[0] = task.G.x;
    goal->as<ob::RealVectorStateSpace::StateType>()->values[1] = task.G.y;
    goal->as<ob::RealVectorStateSpace::StateType>()->values[2] = task.G.z;

    ss.setStartAndGoalStates(start, goal);

    auto planner(std::make_shared<og::BITstar>(ss.getSpaceInformation()));
    ss.setPlanner(planner);

    int64_t start_time;
    bool initial_solution_found = false;
    bool cost_solution_found = false;
    double cost_tag = ratioCost * task.miniCost;

    auto problem_def = ss.getProblemDefinition();
    // 设置路径长度最优目标
    ob::OptimizationObjectivePtr objective(new ob::PathLengthOptimizationObjective(ss.getSpaceInformation()));
    objective->setCostThreshold(ompl::base::Cost(cost_tag));
    problem_def->setOptimizationObjective(objective);
    problem_def->setIntermediateSolutionCallback([&](const ob::Planner *planner, const std::vector<const ob::State*> &states, const ob::Cost &solution_cost) {
        (void) states;
        (void) planner;
        if (!initial_solution_found) {
            ti = utime_ns() - start_time;
            initial_solution_found = true;
            ci = solution_cost.value();
        }

        if (!cost_solution_found && solution_cost.value() < cost_tag) {
            to = utime_ns() - start_time;
            cost_solution_found = true;
        }
    });

    start_time = utime_ns();
    ti = to = uptime * 1e9;
    ob::PlannerStatus solved = ss.solve(uptime);
    printf("%ld, %ld ######\r\n\r\n",ti, to); 
#ifdef udebug
    if (solved) {
        std::cout << "Found solution:" << std::endl;
        ss.simplifySolution();
        const auto &path = ss.getSolutionPath();
        path.print(std::cout);
        std::stringstream namestr;
        namestr << "/home/tzyr/WorkSpace/ompl-1.6.0/img/" << (int64_t)(utime_ns()/1e6) << ".png";
        drawPath(map, path, namestr.str());
    } else {
        std::cout << "No solution found" << std::endl;
    }
#endif
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
    
    BItask taskmap= {{1.5, 1.5, 0.55}, {13.0, 14.0, 13.7}, 48.5};
    statusSpace.boxSpace = {0, 0, 0, 15, 15, 15};
    statusSpace.freeCylinderList.push_back({13.25, 13.25, 4.945, 1.25, 0.11});
    statusSpace.freeCylinderList.push_back({1.75, 1.75, 9.945, 1.25, 0.11});

    statusSpace.obsBoxList.push_back({0, 0, 0, 15, 15, 0.05});
    statusSpace.obsBoxList.push_back({0, 0, 4.95, 15, 15, 0.1});
    statusSpace.obsBoxList.push_back({0, 0, 9.95, 15, 15, 0.1});

    statusSpace.obsSphereList.push_back({3.00, 5.38, 3.00, 1.75});
    statusSpace.obsSphereList.push_back({2.30, 9.89, 1.98, 1.00});
    statusSpace.obsSphereList.push_back({7.34, 3.26, 1.80, 1.25});
    statusSpace.obsSphereList.push_back({6.69, 7.25, 2.80, 0.75});
    statusSpace.obsSphereList.push_back({8.68, 9.72, 2.12, 1.38});
    statusSpace.obsSphereList.push_back({5.76, 12.35, 2.67, 1.50});
    statusSpace.obsSphereList.push_back({12.01, 3.08, 2.88, 2.00});
    statusSpace.obsSphereList.push_back({11.26, 6.50, 3.24, 0.75});
    statusSpace.obsSphereList.push_back({12.30, 10.35, 2.04, 1.25});

    statusSpace.obsCylinderList.push_back({3.30, 1.49, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({1.65, 6.00, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({3.68, 8.30, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({2.74, 11.57, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({5.85, 4.84, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({6.42, 9.42, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({6.98, 12.69, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({8.80, 2.98, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({9.27, 9.27, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({12.15, 2.02, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({10.85, 6.09, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({10.56, 13.00, 5.0, 1.25, 5});
    statusSpace.obsCylinderList.push_back({13.15, 9.32, 5.0, 1.25, 5});

    statusSpace.obsBoxList.push_back({2.40, 1.63, 10 + 1.35, 6.0, 0.8, 3.0});
    statusSpace.obsBoxList.push_back({1.20, 3.70, 10 + 1.40, 3.0, 1.5, 2.2});
    statusSpace.obsBoxList.push_back({2.13, 8.92, 10 + 1.27, 2.7, 2.7, 2.8});
    statusSpace.obsBoxList.push_back({5.91, 5.28, 10 + 0.41, 1.5, 3.0, 3.5});
    statusSpace.obsBoxList.push_back({7.96, 7.50, 10 + 1.88, 1.2, 5.0, 2.9});
    statusSpace.obsBoxList.push_back({4.81, 13.32, 10 + 0.35, 6.0, 1.0, 3.3});
    statusSpace.obsBoxList.push_back({11.16, 2.89, 10 + 1.86, 3.5, 1.0, 2.4});
    statusSpace.obsBoxList.push_back({9.39, 2.13, 10 + 0.10, 1.0, 4.0, 3.1});
    statusSpace.obsBoxList.push_back({10.97, 6.40, 10 + 1.18, 2.5, 2.5, 3.2});
    statusSpace.obsBoxList.push_back({10.82, 10.45, 10 + 0.70, 1.5, 1.5, 2.5});

    int64_t data_ti=0,data_to=0;double data_Ci=0;
    for (size_t i = 0; i < N; i++)
    {
        int64_t ti, to;
        double ci;
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
