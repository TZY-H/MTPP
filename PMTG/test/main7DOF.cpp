#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <queue>

#include "nanoflann.hpp"
#include "PMTG_7DOF.h"
#include "debug3D.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#define rejectPoint3Len (5e-3)
struct uSamplePoint3
{
    double x, y, z;
    double operator%(const uSamplePoint3 &other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return sqrtf32(dx * dx + dy * dy + dz * dz);
    }
    bool operator!=(const uSamplePoint3 &other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return abs(dx) > rejectPoint3Len || abs(dy) > rejectPoint3Len || abs(dz) > rejectPoint3Len;
    }
    bool operator==(const uSamplePoint3 &other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return abs(dx) <= rejectPoint3Len && abs(dy) <= rejectPoint3Len && abs(dz) <= rejectPoint3Len;
    }
    bool operator<(const uSamplePoint3 &other) const
    {
        // // 在这里定义比较规则
        // if (x != other.x)
        //     return x < other.x;
        // return y < other.y;
        double dx = x - other.x;
        if (abs(dx) > rejectPoint3Len)
            return x < other.x;
        double dy = y - other.y;
        if (abs(dy) > rejectPoint3Len)
            return y < other.y;
        double dz = z - other.z;
        if (abs(dz) > rejectPoint3Len)
            return z < other.z;

        return false;
    }
};

// Function to interpolate between two points
Eigen::Vector3d Interpolate(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, double t)
{
    return p1 + t * (p2 - p1);
}

// Function to sample points on a triangle
std::vector<Eigen::Vector3d> SampleTriangle(const Eigen::Vector3d &v0, const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, double density)
{
    std::vector<Eigen::Vector3d> points;

    // Calculate the area of the triangle
    double area = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();

    // Determine the number of samples based on density
    int samples_per_edge = static_cast<int>(std::sqrt(area * density));

    for (int i = 0; i <= samples_per_edge; ++i)
    {
        for (int j = 0; j <= samples_per_edge - i; ++j)
        {
            double u = static_cast<double>(i) / samples_per_edge;
            double v = static_cast<double>(j) / samples_per_edge;
            Eigen::Vector3d point = (1 - u - v) * v0 + u * v1 + v * v2;
            points.push_back(point);
        }
    }
    return points;
}

// Function to convert mesh to point cloud
uint32_t MeshToPointCloud(const aiScene *scene, std::set<uSamplePoint3> &point_cloud, double density)
{
    for (unsigned int index = 0; index < scene->mNumMeshes; ++index)
    {
        const aiMesh *mesh = scene->mMeshes[index];
        for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
        {
            const aiFace &face = mesh->mFaces[i];
            if (face.mNumIndices != 3)
                continue; // Ensure the face is a triangle
            Eigen::Vector3d v0(mesh->mVertices[face.mIndices[0]].x, mesh->mVertices[face.mIndices[0]].y, mesh->mVertices[face.mIndices[0]].z);
            Eigen::Vector3d v1(mesh->mVertices[face.mIndices[1]].x, mesh->mVertices[face.mIndices[1]].y, mesh->mVertices[face.mIndices[1]].z);
            Eigen::Vector3d v2(mesh->mVertices[face.mIndices[2]].x, mesh->mVertices[face.mIndices[2]].y, mesh->mVertices[face.mIndices[2]].z);
            auto sampled_points = SampleTriangle(v0, v1, v2, density);
            for (auto &p : sampled_points)
                point_cloud.insert({p.x(), p.y(), p.z()});
        }
    }
    return point_cloud.size();
}

// Function to draw point cloud
void DrawPointCloud(const std::set<uSamplePoint3> &point_cloud, const GLfloat *color)
{
    glColor3fv(color);
    glBegin(GL_POINTS);
    for (const auto &point : point_cloud)
    {
        glVertex3f(point.x, point.y, point.z);
    }
    glEnd();
}

int main()
{

    statusSpaceStruct statusSpace;
    statusSpace.spatialExtentMax[0] = 2 * M_PI;
    statusSpace.spatialExtentMin[0] = -2 * M_PI;
    statusSpace.spatialExtentMax[1] = 2 * M_PI / 3;
    statusSpace.spatialExtentMin[1] = -2 * M_PI / 3;
    statusSpace.spatialExtentMax[2] = 2 * M_PI;
    statusSpace.spatialExtentMin[2] = -2 * M_PI;
    statusSpace.spatialExtentMax[3] = M_PI * 140.0 / 180.0;
    statusSpace.spatialExtentMin[3] = -M_PI * 140.0 / 180.0;
    statusSpace.spatialExtentMax[4] = M_PI * 2;
    statusSpace.spatialExtentMin[4] = -M_PI * 2;
    statusSpace.spatialExtentMax[5] = M_PI * 120.0 / 180.0;
    statusSpace.spatialExtentMin[5] = -M_PI * 120.0 / 180.0;
    statusSpace.spatialExtentMax[6] = M_PI * 2;
    statusSpace.spatialExtentMin[6] = -M_PI * 2;

    // Assimp import
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile("./test3D/BOX2.STL",
                                             aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    // Assimp import
    Assimp::Importer importer0, importer1, importer2, importer3, importer4, importer5, importer6, importer7, importer8;
    const aiScene *scene0 = importer0.ReadFile("./kortex_description/arms/gen3/7dof/meshes/base_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene1 = importer1.ReadFile("./kortex_description/arms/gen3/7dof/meshes/shoulder_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene2 = importer2.ReadFile("./kortex_description/arms/gen3/7dof/meshes/half_arm_1_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene3 = importer3.ReadFile("./kortex_description/arms/gen3/7dof/meshes/half_arm_2_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene4 = importer4.ReadFile("./kortex_description/arms/gen3/7dof/meshes/forearm_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene5 = importer5.ReadFile("./kortex_description/arms/gen3/7dof/meshes/spherical_wrist_1_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene6 = importer6.ReadFile("./kortex_description/arms/gen3/7dof/meshes/spherical_wrist_2_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene7 = importer7.ReadFile("./kortex_description/arms/gen3/7dof/meshes/bracelet_no_vision_link.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    const aiScene *scene8 = importer8.ReadFile("./kortex_description/2F85_Opened.STL",
                                               aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    double density = 16000 * 1e2;
    std::set<uSamplePoint3> point_cloud;
    MeshToPointCloud(scene, point_cloud, density);
    printf("point_cloud.size(): %ld\r\n", point_cloud.size());
    for (const auto &point : point_cloud)
        statusSpace.ObsTreeCloud.pts.push_back({point.x, point.y, point.z});

    // status testStatusS = {{0, 0.8, 0, 1, 0, 1.4, 0}};
    // status testStatusE = {{-1.4, 1.2, 0, 1.8, 0, -2, 0}};
    status testStatusS = {{-1.4, 0.4, 0, 2.4, 0.2, -1.0, -1.6}};
    status testStatusE = {{0.2, 1.0, 0, 0.8, 2.0, -1.6, -2.8}};

    PMTG7DOF PMTGR3test;
    PMTGR3test.InitializeInvironment(statusSpace);
    PMTGR3test.InitializeSubTree(30, 260000);

    if (!(PMTGR3test.ValidityCheck(testStatusS) && PMTGR3test.ValidityCheck(testStatusE)))
    {
        printf("ValidityCheck !!!\r\n");
        return 0;
    }
    PMTGR3test.SupplementaryRun();
    printf("NodeMap.size(): %ld\r\n", PMTGR3test.NodeMap.size());
    PMTGR3test.MasterTaskWait(200000);

    // for (int32_t i = 0; i < 30; i++)
    //     std::cout << "i: " << i << ", linknum:" << PMTGR3test.treeLinkMap[i].size() << std::endl;

    std::list<status> Path;
    int64_t tt = PMTGR3test.utime_ns();
    double AStarCost = PMTGR3test.GetSubTreeGraphPath(testStatusS,
                                                     testStatusE, Path);
    std::cout << "time: " << PMTGR3test.utime_ns() - tt << ", AStarCost: " << AStarCost << std::endl;
    if (Path.size() == 0)
        return 0;
    printf("Path.size(): %ld\r\n", Path.size());
    PMTGR3test.showPath(Path);

    std::list<status> Path2;
    tt = PMTGR3test.utime_ns();
    double OptCost = PMTGR3test.GetRGGOptPath(testStatusS,
                                             testStatusE, Path2, AStarCost);
    std::cout << "time: " << PMTGR3test.utime_ns() - tt << ", OptCost: " << OptCost << std::endl;
    printf("Path2.size(): %ld\r\n", Path2.size());
    PMTGR3test.showPath(Path2);

    // 创建一个窗口，设置窗口大小为640x480
    pangolin::CreateWindowAndBind("Example", 640, 480);

    // 启动深度测试
    glEnable(GL_DEPTH_TEST);
    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Set light properties
    GLfloat light_pos[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat light_diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    GLfloat light_specular[] = {0.001f, 0.001f, 0.001f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    // 创建一个视图
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin::ModelViewLookAt(1.5, 1.5, 1.5, 0, 0, 0, pangolin::AxisZ));

    // 创建一个交互面板
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);
    Eigen::Matrix4d To;
    To << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    while (!pangolin::ShouldQuit())
    {
        // 清除屏幕和深度缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // 设置模型视图矩阵
        glLoadIdentity();
        d_cam.Activate(s_cam);
        GLfloat colorBASE[] = {0.65, 0.65, 0.65, 1.0};
        GLfloat colorR[] = {1.0, 0.0, 0.0, 1.0};

        {
            if (Path.size() > 1)
                Path.pop_front();
            else if (Path2.size() > 3)
                Path.splice(Path.end(), Path2);
            status x = Path.front();
            VectorXd Q(7);
            Q << x.q[0], x.q[1], x.q[2], x.q[3], x.q[4], x.q[5], x.q[6];
            PMTGR3test.Gen3obj.Gen3_UpBtoH(Q, true);
            GLfloat colorBASER[] = {0.5, 0.5, 0.5, 1.0};
            // if (!PMTGR3test.ValidityCheck(x))
            // {
            //     colorBASER[0] = 0;
            //     colorBASER[1] = 0;
            //     colorBASER[1] = 1;
            //     printf("ValidityCheck!!!\r\n");
            // }
            //     for (int32_t i = 0; i < PMTGR3test.Gen3obj.Gen3_TMat.size(); i++)
            //     {
            //         Vector3d p0, p1;
            //         if (i != 0)
            //         {
            //             auto &TM0 = PMTGR3test.Gen3obj.Gen3_TMat[i - 1];
            //             p0 = TM0.block<3, 1>(0, 3);
            //         }
            //         else
            //             p0 = {0, 0, 0};
            //         auto &TM1 = PMTGR3test.Gen3obj.Gen3_TMat[i];
            //         p1 = TM1.block<3, 1>(0, 3);
            //         int32_t n = (p0 - p1).norm() / 0.06 + 1;
            //         for (int32_t j = 0; j <= n; j++)
            //         {
            //             double t = (double)j / n;
            //             Vector3d p = t * p1 + (1 - t) * p0;
            //             if (j)
            //                 drawSphere(p[0], p[1], p[2], 0.050, colorBASER, false);
            //             else
            //                 drawSphere(p[0], p[1], p[2], 0.055, colorR, false);
            //         }
            //     }
        }
        bool wireframe = false;
        GLfloat color[] = {0.8f, 0.8f, 0.8f, 1.0f}; // Red color
        DrawMesh(scene0, wireframe, color, To);
        DrawMesh(scene1, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[0]);
        DrawMesh(scene2, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[1]);
        DrawMesh(scene3, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[2]);
        DrawMesh(scene4, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[3]);
        DrawMesh(scene5, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[4]);
        DrawMesh(scene6, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[5]);
        DrawMesh(scene7, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[6]);
        DrawMesh(scene8, wireframe, color, PMTGR3test.Gen3obj.Gen3_TMat[7]);
        // DrawPointCloud(point_cloud, colorR);
        Eigen::Matrix4d T;
        T << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
        DrawMesh(scene, true, colorR, T);
        // DrawMesh(scene, false, colorBASE, T);

        // drawSphere(0,0,0, 0.055, colorR, true);
        // 交换缓冲区并检查事件
        // usleep();

        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow("Example");
    printf("!!!s!!!@@@ \r\n");

    return 0;
}