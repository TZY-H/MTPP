
#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
#include <list>

#include "nanoflann.hpp"
#include "uGen3Ctrl.h"
#include "statusSpace7DOF.h"

uGen3Ctrl Gen3obj;
nanoflann::PointCloud_d<double, 3> ObsTreeCloud;
kd_tree_3 obsSearchTree(3, ObsTreeCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

double distanceBetweenSegments(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector3d &D)
{
    Vector3d u = B - A;
    Vector3d v = D - C;
    Vector3d w = A - C;

    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w);
    double e = v.dot(w);

    double DD = a * c - b * b;
    double sc, sN, sD = DD;
    double tc, tN, tD = DD;

    if (DD < 1e-6)
    {sN = 0.0;sD = 1.0;tN = e;tD = c;}
    else
    {
        sN = (b * e - c * d);tN = (a * e - b * d);
        if (sN < 0.0){sN = 0.0;tN = e;tD = c;}
        else if (sN > sD){sN = sD;tN = e + b;tD = c;}
    }

    if (tN < 0.0)
    {
        tN = 0.0;
        if (-d < 0.0) {sN = 0.0;}
        else if (-d > a) {sN = sD;}
        else {sN = -d;sD = a;}
    }
    else if (tN > tD)
    {
        tN = tD;
        if ((-d + b) < 0.0) {sN = 0;}
        else if ((-d + b) > a){sN = sD;}
        else {sN = (-d + b);sD = a;}
    }
    sc = (fabs(sN) < 1e-6 ? 0.0 : sN / sD);
    tc = (fabs(tN) < 1e-6 ? 0.0 : tN / tD);
    Vector3d dP = w + (sc * u) - (tc * v);
    return dP.norm();
}

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
