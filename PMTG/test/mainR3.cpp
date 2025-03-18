#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <queue>

#include "PMTG_3D.h"
#include "debug3D.h"

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
    // srand((unsigned)time(NULL)); // 初始化随机数种子

    cv::namedWindow("Video", 0);
    cv::resizeWindow("Video", 1200, 1200);
    cv::setMouseCallback("Video", onMouse, NULL);

    statusSpaceStruct statusSpace;
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

    status testStatusS = {1.5, 1.5, 0.55};
    status testStatusE = {13.0, 14.0, 13.7};

    PMTGR3 PMTGR3test;
    PMTGR3test.InitializeInvironment(statusSpace);
    PMTGR3test.InitializeSubTree(30, 60000);
    PMTGR3test.SupplementaryRun();
    PMTGR3test.MasterTaskWait(2000);
    std::list<status> Path;
    int64_t tt = PMTGR3test.utime_ns();
    double AStarCost = PMTGR3test.GetSubTreeGraphPath(testStatusS, testStatusE, Path);
    std::cout << "time: " << PMTGR3test.utime_ns() - tt << ", AStarCost: " << AStarCost << std::endl;

    tt = PMTGR3test.utime_ns();
    std::list<status> Path2;
    double OptCost = PMTGR3test.GetRGGOptPath(testStatusS, testStatusE, Path2, AStarCost);
    std::cout << "time: " << PMTGR3test.utime_ns() - tt << ", OptCost: " << OptCost << std::endl;

    // 创建一个窗口，设置窗口大小为640x480
    pangolin::CreateWindowAndBind("Example", 640, 480);

    // 启动深度测试
    glEnable(GL_DEPTH_TEST);

    // 创建一个视图
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin::ModelViewLookAt(30, 30, 30, 0, 0, 0, pangolin::AxisZ));

    // 创建一个交互面板
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit())
    {
        // 清除屏幕和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // 设置模型视图矩阵
        glLoadIdentity();
        d_cam.Activate(s_cam);
        GLfloat colorBASE[] = {0.5, 0.5, 0.5, 1.0};
        drawCube(PMTGR3test.statusSpace.boxSpace, colorBASE, true);

        GLfloat colorF[] = {0.9, 0.3, 0.3, 1.0};
        GLfloat colorH[] = {0.2, 0.2, 0.2, 1.0};
        int i =0;
        for (const auto &space : PMTGR3test.statusSpace.obsBoxList)
        {
            if (i++<3)
                drawCube(space, colorH, false);
            else
                drawCube(space, colorF, false);
            drawCube(space, colorF, true);
        }

        GLfloat colorI[] = {1.0, 0.7, 0.28, 1.0};
        for (const auto &space : PMTGR3test.statusSpace.obsCylinderList)
        {
            drawCylinder(space, colorI, false);
            drawCylinder(space, colorBASE, true);
        }

        GLfloat colorO[] = {0.66, 0.78, 0.88, 1.0};
        for (const auto &space : PMTGR3test.statusSpace.obsSphereList)
        {
            drawSphere(space, colorO, false);
            drawSphere(space, colorBASE, true);
        }

        GLfloat colorFI[] = {0.2, 0.7, 0.7, 1.0};
        for (const auto &space : PMTGR3test.statusSpace.freeCylinderList)
        {
            drawCylinder(space, colorFI, false);
            // drawCylinder(space, colorBASE, true);
        }

        GLfloat colorS[] = {1.0, 0.0, 0.0, 1.0};
        GLfloat colorE[] = {0.0, 0.0, 1.0, 1.0};
        GLfloat colorT[] = {1.0, 0.5, 0.95, 1.0};
        drawSphere(testStatusS.x, testStatusS.y, testStatusS.z, 0.2, colorS, false);
        drawSphere(testStatusE.x, testStatusE.y, testStatusE.z, 0.2, colorE, false);
        for (const auto &subtree : PMTGR3test.treelist)
        {
            drawSphere(subtree.treeRoot.x, subtree.treeRoot.y, subtree.treeRoot.z, 0.2, colorT, false);
        }

        DrawPath(Path, colorE);
        // 交换缓冲区并检查事件
        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow("Example");
    printf("!!!s!!!@@@ \r\n");

    return 0;
}