#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <queue>
#include "PMTG_SO2.h"

std::mutex mtx; // 创建一个互斥锁
// 回调函数，当鼠标左键按下时调用

int32_t updata = false;
double MouseX, MouseY;
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        updata = true;
        MouseX = x;
        MouseY = y;
        // 在控制台输出鼠标左击位置的坐标
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
}

int main()
{
    srand((unsigned)(181651)); // 初始化随机数种子

    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse, NULL);
    std::vector<cv::Scalar> treeColors;
    std::vector<status> objectPointSet;
    for (double r = 1.0; r < 2.0 * 3.1415 - 1.0; r += 0.214)
    {
        double sinth = sin(-r);
        double costh = cos(-r);
        // objectPointSet.push_back({56.0 * costh, 56.0 * sinth, 0});
        objectPointSet.push_back({72.0 * costh, 72.0 * sinth, 0});
    }
    for (double r = 1.0; r < 2.0 * 3.1415 - 1.0; r += 0.214)
    {
        double sinth = sin(r);
        double costh = cos(r);
        objectPointSet.push_back({56.0 * costh, 56.0 * sinth, 0});
        // objectPointSet.push_back({72.0 * costh, 72.0 * sinth, 0});
    }
    status testStatus = {390, 415, 0};
    status testStatusS = {390, 415, 0};
    status testStatusE = {1571, 1571, 3.1415926};
    PMTGSE2 PMTGSE2test;
    PMTGSE2test.InitializeInvironment("./map/FMTmapSE2.png", 127, 3.1415926 * 3);
    PMTGSE2test.objectPointSet = objectPointSet;
    PMTGSE2test.InitializeSubTree(30, 60000);

    cv::Mat Freemap = PMTGSE2test.Freemap.clone();
    for (const auto &subtree : PMTGSE2test.treelist)
    {
        double costh = cos(subtree.treeRoot.r);
        double sinth = sin(subtree.treeRoot.r);
        cv::Scalar color;
        if (PMTGSE2test.ValidityCheck(subtree.treeRoot))
        {
            double blue = 100 + rand() % 128;
            double green = 100 + rand() % 128;
            double red = 80 + rand() % 128;
            color = cv::Scalar(blue, green, red);
        }
        else
            color = cv::Scalar(0, 0, 255);
        double px_old = costh * PMTGSE2test.objectPointSet.back().x - sinth * PMTGSE2test.objectPointSet.back().y + subtree.treeRoot.x;
        double py_old = sinth * PMTGSE2test.objectPointSet.back().x + costh * PMTGSE2test.objectPointSet.back().y + subtree.treeRoot.y;
        for (const status &p : PMTGSE2test.objectPointSet)
        {
            double px = costh * p.x - sinth * p.y + subtree.treeRoot.x;
            double py = sinth * p.x + costh * p.y + subtree.treeRoot.y;
            cv::Point pt1(static_cast<int>(px), static_cast<int>(py));
            // cv::circle(Freemap, pt1, 3, color, -1);
            cv::Point pt2(static_cast<int>(px_old), static_cast<int>(py_old));
            cv::line(Freemap, pt1, pt2, color, 3);
            px_old = px;
            py_old = py;
        }
    }
    cv::imshow("Video", Freemap);
    cv::waitKey();
    PMTGSE2test.SupplementaryRun();
    std::cout << "NodeSize: " << PMTGSE2test.NodeMap.size() << std::endl;
    PMTGSE2test.MasterTaskWait(2000);
    std::list<status> Path;
    int64_t tt = PMTGSE2test.utime_ns();
    double AStarCost = PMTGSE2test.GetSubTreeGraphPath(testStatusS, testStatusE, Path);
    std::cout << "time: " << PMTGSE2test.utime_ns() - tt << ", AStarCost: " << AStarCost << std::endl;
    if (AStarCost > doubleConserMAX)
        return 0;

    PMTGSE2test.showPath(Path, Freemap);
    cv::waitKey();
    
    tt = PMTGSE2test.utime_ns();
    double OptCost = PMTGSE2test.GetRGGOptPath(testStatusS, testStatusE, Path, AStarCost);
    std::cout << "time: " << PMTGSE2test.utime_ns() - tt << ", OptCost: " << OptCost << std::endl;
    PMTGSE2test.showPath(Path, PMTGSE2test.Freemap);
    cv::waitKey();
    printf("!!!s!!!@@@ \r\n");
    return 0;
}
