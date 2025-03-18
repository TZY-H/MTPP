#ifndef __DEBUG3D_H
#define __DEBUG3D_H
#include <pangolin/pangolin.h>

// #include "uGen3Ctrl.h"
#include "rrf3D.h"
// #include "PMTG_7DOF.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

void drawCylinder(double x, double y, double z, double r, double h, const GLfloat *color, bool drawWireframe = false);
void drawSphere(double x, double y, double z, double r, const GLfloat *color, bool drawWireframe = false);
void drawCube(double x, double y, double z, double lenx, double leny, double lenz, const GLfloat *color, bool drawWireframe = false);
void drawCylinder(const CylinderSpace &space, const GLfloat *color, bool drawWireframe = false);
void drawSphere(const SphereSpace &space, const GLfloat *color, bool drawWireframe = false);
void drawCube(const BoxSpace &space, const GLfloat *color, bool drawWireframe = false);
void DrawMesh(const aiScene *scene, bool wireframe, const GLfloat *color, const Eigen::Matrix4d &T);

void drawPoints(const std::vector<Eigen::Vector3d> &points);
void generateBoxSurfacePoints(double lx, double ly, double lz, double c, std::vector<Eigen::Vector3d> &points);

void drawLine(const status &p1, const status &p2, const GLfloat *color);
void DrawPath(const std::list<status> &path, const GLfloat *color);

#endif