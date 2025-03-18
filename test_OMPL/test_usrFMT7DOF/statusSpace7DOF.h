#ifndef __statusSpace3D_H
#define __statusSpace3D_H
#include <iostream>
#include <stdio.h>
#include <queue>
#include <deque>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "nanoflann.hpp"
#include "uGen3Ctrl.h"

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 3>>,
    nanoflann::PointCloud_d<double, 3>, 3>
    kd_tree_3;
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, nanoflann::PointCloud_d<double, 7>>,
    nanoflann::PointCloud_d<double, 7>, 7>
    kd_tree_7;
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
double distanceBetweenSegments(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector3d &D);
uint32_t MeshToPointCloud(const aiScene *scene, std::set<uSamplePoint3> &point_cloud, double density);
struct BIpoint
{
    double q[7];
};
struct BItask
{
    BIpoint S;
    BIpoint G;
    double miniCost;
};

extern uGen3Ctrl Gen3obj;
extern nanoflann::PointCloud_d<double, 3> ObsTreeCloud;
extern kd_tree_3 obsSearchTree;

#endif