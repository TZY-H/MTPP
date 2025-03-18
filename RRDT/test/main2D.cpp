#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
#include <thread>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <thread>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

#include "rrf.h"
#define ratioCost 1.02

// 打印 savedata_to
void printSavedataTo(const std::vector<int64_t> &savedata)
{
    for (const auto &value : savedata)
    {
        std::cout << value << ", ";
    }
    std::cout << "\n";
}

int main()
{
    cv::namedWindow("showMap", cv::WINDOW_NORMAL);
    cv::resizeWindow("showMap", 1400, 1400);

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义分布范围（整个 int32_t 范围）
    std::uniform_int_distribution<int32_t> distrib(INT32_MIN, INT32_MAX);
    int32_t rr = distrib(gen);
    printf("rr %d\r\n", rr);
    
    RRFplanner planner(rr);
    planner.InitializeInvironment("./map/MAZE4.png", 128);

    planner.E_distance = 50;
    status S = {190, 3080};
    status E = {3100, 180};
    std::list<status> path;

    std::vector<int64_t> savedata_ti;
    std::vector<int64_t> savedata_to;
    int64_t data_ti = 0, data_to = 0, N = 10;
    double data_Ci = 0;
    for (size_t i = 0; i < N; i++)
    {
        printf(" i: %ld\r\n", i);

        // long t0 = utime_ns();
        double cost = planner.planningTask(S, E, path, ratioCost * 4656, 999999);
        std::cout << planner.t_init << std::endl;
        std::cout << planner.t_opt << std::endl;
        savedata_ti.push_back(planner.t_init);
        savedata_to.push_back(planner.t_opt);
        data_ti += planner.t_init;
        data_to += planner.t_opt;
        data_Ci += planner.c_init;

        cv::Mat showMap = planner.Freemap.clone();
        if (path.size() >= 2)
        {
            status point0 = path.front();
            for (status point1 : path)
            {
                cv::Point startPoint((int)point0.q[0], (int)point0.q[1]);
                cv::Point endPoint((int)point1.q[0], (int)point1.q[1]);
                cv::line(showMap, startPoint, endPoint, cv::Scalar(0, 0, 255), 2);
                point0 = point1;
            }
        }
        printf("cost:%lf\r\n", cost);
        cv::imshow("showMap", showMap);
        cv::waitKey(30);
    }
    data_ti = data_ti / N;
    data_to = data_to / N;
    data_Ci = data_Ci / N / 4656;
    std::cout << "savedata_ti: " << std::endl;
    printSavedataTo(savedata_ti);
    std::cout << "savedata_to: " << std::endl;
    printSavedataTo(savedata_to);
    std::cout << "data_ti: " << data_ti/1e6 << "  data_to: " << data_to/1e6 << "  data_Ci: " << data_Ci << std::endl;

    return 0;
}
