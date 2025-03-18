#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
#include <thread>
#include <random>

#include <thread>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

#include "rrf3D.h"
#include "debug3D.h"
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

int8_t pkey=1;
void OnKeyPressW() {
    std::cout << "Key 'W' pressed." << std::endl;
    pkey = 0;
}

int main()
{
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

    status S = {1.5, 1.5, 0.55};
    status E = {13.0, 14.0, 13.7};

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义分布范围（整个 int32_t 范围）
    std::uniform_int_distribution<int32_t> distrib(INT32_MIN, INT32_MAX);
    int32_t rr = distrib(gen);
    printf("rr %d\r\n", rr);
    
    RRFplanner planner(rr);
    planner.InitializeInvironment(statusSpace);

    planner.E_distance = 0.9;
    std::list<status> path;

    std::vector<int64_t> savedata_ti;
    std::vector<int64_t> savedata_to;
    int64_t data_ti = 0, data_to = 0, N = 10;
    double data_Ci = 0;

        // 创建一个窗口，设置窗口大小为640x480
        pangolin::CreateWindowAndBind("Example", 640, 480);
        // 注册按键回调函数
        pangolin::RegisterKeyPressCallback('w', OnKeyPressW); // 为 'w' 键注册回调
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
    
    for (size_t i = 0; i < N; i++)
    {
        printf(" i: %ld\r\n", i);

        // long t0 = utime_ns();
        double cost = planner.planningTask(S, E, path, ratioCost * 48.5, 999999);
        std::cout << planner.t_init << std::endl;
        std::cout << planner.t_opt << std::endl;
        std::cout << path.size() << std::endl;
        std::cout << cost << std::endl;
        savedata_ti.push_back(planner.t_init);
        savedata_to.push_back(planner.t_opt);
        data_ti += planner.t_init;
        data_to += planner.t_opt;
        data_Ci += planner.c_init;

        while (pkey)
        {
            // 清除屏幕和深度缓冲区
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // 设置模型视图矩阵
            glLoadIdentity();
            d_cam.Activate(s_cam);
            GLfloat colorBASE[] = {0.5, 0.5, 0.5, 1.0};
            drawCube(planner.statusSpace.boxSpace, colorBASE, true);

            GLfloat colorF[] = {0.9, 0.3, 0.3, 1.0};
            GLfloat colorH[] = {0.2, 0.2, 0.2, 1.0};
            int i = 0;
            for (const auto &space : planner.statusSpace.obsBoxList)
            {
                if (i++ < 3)
                    drawCube(space, colorH, false);
                else
                    drawCube(space, colorF, false);
                drawCube(space, colorF, true);
            }

            GLfloat colorI[] = {1.0, 0.7, 0.28, 1.0};
            for (const auto &space : planner.statusSpace.obsCylinderList)
            {
                drawCylinder(space, colorI, false);
                drawCylinder(space, colorBASE, true);
            }

            GLfloat colorO[] = {0.66, 0.78, 0.88, 1.0};
            for (const auto &space : planner.statusSpace.obsSphereList)
            {
                drawSphere(space, colorO, false);
                drawSphere(space, colorBASE, true);
            }

            GLfloat colorFI[] = {0.2, 0.7, 0.7, 1.0};
            for (const auto &space : planner.statusSpace.freeCylinderList)
            {
                drawCylinder(space, colorFI, false);
                // drawCylinder(space, colorBASE, true);
            }

            GLfloat colorS[] = {1.0, 0.0, 0.0, 1.0};
            GLfloat colorE[] = {0.0, 0.0, 1.0, 1.0};
            GLfloat colorT[] = {1.0, 0.5, 0.95, 1.0};
            drawSphere(S.q[0], S.q[1], S.q[2], 0.2, colorS, false);
            drawSphere(E.q[0], E.q[1], E.q[2], 0.2, colorE, false);

            DrawPath(path, colorE);
            // 交换缓冲区并检查事件
            pangolin::FinishFrame();
        }

    }
    data_ti = data_ti / N;
    data_to = data_to / N;
    data_Ci = data_Ci / N / 48.5;
    std::cout << "savedata_ti: " << std::endl;
    printSavedataTo(savedata_ti);
    std::cout << "savedata_to: " << std::endl;
    printSavedataTo(savedata_to);
    std::cout << "data_ti: " << data_ti/1e6 << "  data_to: " << data_to/1e6 << "  data_Ci: " << data_Ci << std::endl;
    pangolin::DestroyWindow("Example");
    return 0;
}
