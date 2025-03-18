#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/base/ProblemDefinition.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"

// #define udebug
#define uptime 25.0
#define ratioCost 1.02

namespace ob = ompl::base;
namespace og = ompl::geometric;
struct BIpoint
{
    double x;
    double y;
};
struct BItask
{
    BIpoint S;
    BIpoint G;
    double miniCost;
    std::string Path;
};


BIpoint mousePoint = {-1, -1};
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    (void) userdata;
    // cv::Mat* bimap = (cv::Mat*) userdata;
    if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
    {
        mousePoint.x = x;
        mousePoint.y = y;
        // 在控制台输出鼠标左击位置的坐标
        // std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
        // std::cout << "BIimap: " << (int)(bimap->at<uint8_t>(y, x)) << std::endl;
    }
}
void drawPath(const cv::Mat &map, const ompl::geometric::PathGeometric &path, const std::string &filename) {
    cv::Mat result;
    cv::cvtColor(map, result, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < path.getStateCount() - 1; ++i) {
        const auto *state1 = path.getState(i)->as<ompl::base::RealVectorStateSpace::StateType>();
        const auto *state2 = path.getState(i + 1)->as<ompl::base::RealVectorStateSpace::StateType>();

        cv::Point pt1(static_cast<int>(state1->values[0]), static_cast<int>(state1->values[1]));
        cv::Point pt2(static_cast<int>(state2->values[0]), static_cast<int>(state2->values[1]));

        cv::line(result, pt1, pt2, cv::Scalar(0, 0, 255), 2);
    }
    if (!cv::imwrite(filename, result)) {
        std::cerr << "Failed to save the image: " << filename << std::endl;
    } else {
        std::cout << "Path saved to " << filename << std::endl;
    }
    cv::imshow("Video",result);
    cv::waitKey(500);
}

cv::Mat loadMap(const std::string &filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load map: " << filename << std::endl;
        exit(1);
    }
    return image;
}

bool isStateValid(const ob::State *state, const cv::Mat &map) {
    const ob::RealVectorStateSpace::StateType *rvState = state->as<ob::RealVectorStateSpace::StateType>();
    int x = static_cast<int>(rvState->values[0]);
    int y = static_cast<int>(rvState->values[1]);

    if (x < 0 || y < 0 || x >= map.cols || y >= map.rows) {
        return false;
    }

    return map.at<uchar>(y, x) > 127; // Assuming white pixels are free space
}

void plan(const cv::Mat &map,const BItask &task, int64_t &ti, int64_t &to, double &ci) 
{
    auto space(std::make_shared<ob::RealVectorStateSpace>(2));
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0);
    bounds.setHigh(0, map.cols);
    bounds.setHigh(1, map.rows);
    space->setBounds(bounds);

    og::SimpleSetup ss(space); 
    ss.setStateValidityChecker([&](const ob::State *state) {
        return isStateValid(state, map);
    });
    ss.getSpaceInformation()->setStateValidityCheckingResolution(2.0/std::max(map.cols,map.rows));

    ob::ScopedState<> start(space);
    start->as<ob::RealVectorStateSpace::StateType>()->values[0] = task.S.x;
    start->as<ob::RealVectorStateSpace::StateType>()->values[1] = task.S.y;

    ob::ScopedState<> goal(space);
    goal->as<ob::RealVectorStateSpace::StateType>()->values[0] = task.G.x;
    goal->as<ob::RealVectorStateSpace::StateType>()->values[1] = task.G.y;

    ss.setStartAndGoalStates(start, goal);

    auto planner(std::make_shared<og::ABITstar>(ss.getSpaceInformation()));
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
    
    BItask taskmap= {{190, 3080}, {3100, 180}, 4656, "./test_usrFMT2D/MAZE4.png"};

    int64_t data_ti=0,data_to=0;
    double data_Ci=0;
    for (size_t i = 0; i < N; i++)
    {
        cv::Mat map = loadMap(taskmap.Path);
        int64_t ti, to;
        double ci;
        plan(map,taskmap, ti, to, ci);
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

    std::cout << "savedata_ti: " <<std::endl; 
    printSavedataTo(savedata_ti);
    std::cout << "savedata_to: " <<std::endl; 
    printSavedataTo(savedata_to);
    
    std::cout << "data_ti: " << data_ti/1e6 << "  data_to: " << data_to/1e6 << "  data_Ci: " << data_Ci << std::endl;
    return 0;
}
