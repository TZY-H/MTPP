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

struct BoxSpace
{
    double x, y, z;
    double lenx, leny, lenz;
};
struct SphereSpace
{
    double x, y, z, r;
};
struct CylinderSpace
{
    double x, y, z, r, h;
};

struct statusSpaceStruct
{
    BoxSpace boxSpace;

    std::vector<BoxSpace> obsBoxList;
    std::vector<SphereSpace> obsSphereList;
    std::vector<CylinderSpace> obsCylinderList;

    std::vector<BoxSpace> freeBoxList;
    std::vector<SphereSpace> freeSphereList;
    std::vector<CylinderSpace> freeCylinderList;
};



#endif