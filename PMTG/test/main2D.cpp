#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <queue>
#include "PMTG_2D.h"

std::mutex mtx; // 创建一个互斥锁
// 回调函数，当鼠标左键按下时调用

int32_t updata = false;

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        updata = true;
        // 在控制台输出鼠标左击位置的坐标
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
    }
}
// Left button of the mouse is clicked - position (1822, 1520)
// Left button of the mouse is clicked - position (643, 1894)707
// Left button of the mouse is clicked - position (1067, 1224)
// Left button of the mouse is clicked - position (85, 1439)
// Left button of the mouse is clicked - position (919, 285)

int main()
{
    srand((unsigned)12564);
    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse, NULL);
    std::vector<cv::Scalar> treeColors;

    PMTG2D mft2dtest;
    mft2dtest.InitializeInvironment("./map/MAZE4.png", 127);
    mft2dtest.InitializeSubTree(64, 60000);
    for (int32_t i = 0; i < mft2dtest.treelist.size(); i++)
    {
        double blue = 100 + rand() % 128;
        double green = 100 + rand() % 128;
        double red = 80 + rand() % 128;
        treeColors.push_back({blue, green, red});
    }
    for (size_t runCount = 0; runCount < 1; runCount++)
    {
        mft2dtest.SupplementaryRun();
        mft2dtest.MasterTaskWait(2000);

        // mft2dtest.stopSubTask();
        cv::Mat Freemap = mft2dtest.Freemap.clone();
        for (const auto &pair : mft2dtest.NodeMap)
        {
            status ptS = pair.first;
            status ptP = pair.second.par;
            if (ptP.x < 0)
            {
                cv::Point pt1(static_cast<int>(ptS.x), static_cast<int>(ptS.y));
                cv::circle(Freemap, pt1, 3, cv::Scalar(255, 0, 0), -1);
                continue;
            }
            cv::Point pt1(static_cast<int>(ptS.x), static_cast<int>(ptS.y));
            cv::Point pt2(static_cast<int>(ptP.x), static_cast<int>(ptP.y));
            cv::line(Freemap, pt1, pt2, treeColors[pair.second.tree], 2);
        }
        for (auto &tree : mft2dtest.treelist)
        {
            int px = static_cast<int>(tree.treeRoot.x);
            int py = static_cast<int>(tree.treeRoot.y);
            cv::circle(Freemap, cv::Point(px, py), 12, cv::Scalar(0, 0, 255), -1);
            std::ostringstream text;
            text << tree.treeIndex;
            cv::putText(Freemap, text.str(), cv::Point(px, py), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 0, 255), 3);
        }
        for (int32_t subtreeIndex = 0; subtreeIndex < mft2dtest.treeLinkMap.size(); subtreeIndex++)
        {
            auto &links = mft2dtest.treeLinkMap[subtreeIndex];
            for (auto &link : links)
            {
                status ptS = link.second.second;
                treelink_concurrentMap::accessor accXnearInSelfTree;
                mft2dtest.treeLinkMap[link.first].find(accXnearInSelfTree, subtreeIndex);
                status ptP = accXnearInSelfTree->second.second;

                cv::Point pt1(static_cast<int>(ptS.x), static_cast<int>(ptS.y));
                cv::Point pt2(static_cast<int>(ptP.x), static_cast<int>(ptP.y));
                cv::line(Freemap, pt1, pt2, cv::Scalar(0, 0, 255), 4);
            }
        }
        cv::imwrite("test2dPMTG.png",Freemap);
        status S = {190, 3080};
        status E = {3100, 180};
        cv::circle(Freemap, cv::Point(S.x, S.y), 12, cv::Scalar(255, 0, 0), -1);
        cv::circle(Freemap, cv::Point(E.x, E.y), 12, cv::Scalar(255, 0, 0), -1);
        cv::imshow("Video", Freemap);
        if (cv::waitKey(00) == 'q')
            break;
        std::list<status> Path;
        int64_t tt = mft2dtest.utime_ns();
        double AStarCost = mft2dtest.GetSubTreeGraphPath(S, E, Path);
        std::cout << "time: " << mft2dtest.utime_ns() - tt << ", AStarCost: " << AStarCost << std::endl;
        status x_old = Path.front();
        for (const auto &x : Path)
        {
            cv::Point pt1(static_cast<int>(x_old.x), static_cast<int>(x_old.y));
            cv::Point pt2(static_cast<int>(x.x), static_cast<int>(x.y));
            cv::line(Freemap, pt1, pt2, cv::Scalar(0, 0, 255), 10);
            x_old = x;
        }
        cv::imwrite("test2dINIT.png",Freemap);
        cv::imshow("Video", Freemap);
        if (cv::waitKey(00) == 'q')
            break;
        tt = mft2dtest.utime_ns();
        double OptCost = mft2dtest.GetRGGOptPath(S, E, Path, AStarCost);
        std::cout << "time: " << mft2dtest.utime_ns() - tt << ", OptCost: " << OptCost << std::endl;
        Freemap = mft2dtest.Freemap.clone();
        for (const auto &pair : mft2dtest.RGGOptNodeMap)
        {
            status ptS = pair.first;
            status ptP = pair.second.par;
            if (ptP.x < 0)
            {
                double dx1 = ptS.x - S.x;
                double dy1 = ptS.y - S.y;
                double dx2 = ptS.x - E.x;
                double dy2 = ptS.y - E.y;
                if((sqrt(dx1*dx1 + dy1*dy1)+sqrt(dx2*dx2 + dy2*dy2))<OptCost)
                {
                    cv::Point pt1(static_cast<int>(ptS.x), static_cast<int>(ptS.y));
                    cv::circle(Freemap, pt1, 3, cv::Scalar(255, 0, 0), -1);
                }
                continue;
            }
            cv::Point pt1(static_cast<int>(ptS.x), static_cast<int>(ptS.y));
            cv::Point pt2(static_cast<int>(ptP.x), static_cast<int>(ptP.y));
            cv::line(Freemap, pt1, pt2, treeColors[pair.second.tree], 2);
        }
        cv::imshow("Video", Freemap);
        if (cv::waitKey(00) == 'q')
            break;
        x_old = Path.front();
        for (const auto &x : Path)
        {
            cv::Point pt1(static_cast<int>(x_old.x), static_cast<int>(x_old.y));
            cv::Point pt2(static_cast<int>(x.x), static_cast<int>(x.y));
            cv::line(Freemap, pt1, pt2, cv::Scalar(0, 0, 255), 10);
            x_old = x;
            cv::imshow("Video", Freemap);
            cv::waitKey(20);
        }
        cv::imwrite("test2dOPT.png",Freemap);
        cv::imshow("Video", Freemap);
        if (cv::waitKey(00) == 'q')
            break;
    }

    mft2dtest.stopSubTask();
    printf("!!!s!!!@@@ \r\n");
    // cv::imwrite("./map/PMTG2DLOL_20.png", mft2dtest.Freemap);
    return 0;
}
