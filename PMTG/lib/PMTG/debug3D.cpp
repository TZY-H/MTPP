#include "debug3D.h"

void drawCylinder(double x, double y, double z, double r, double h, const GLfloat *color, bool drawWireframe)
{
    const int sides = 30; // 圆柱侧面的分段数
    if (drawWireframe)
    {
        glColor3fv(color); // 设置线框颜色
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        glColor3fv(color); // 设置实体颜色
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    // Set material color
    GLfloat material_specular[] = {5e-7f, 5e-7f, 5e-7f, 1.0f}; // 镜面反射颜色
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0f);

    // 绘制底面圆
    glBegin(GL_TRIANGLE_FAN);
    glVertex3d(x, y, z); // 中心点
    for (int i = 0; i <= sides; ++i)
    {
        double angle = 2.0 * M_PI * i / sides;
        glVertex3d(x + r * cos(angle), y + r * sin(angle), z);
    }
    glEnd();

    // 绘制顶面圆
    glBegin(GL_TRIANGLE_FAN);
    glVertex3d(x, y, z + h); // 中心点
    for (int i = 0; i <= sides; ++i)
    {
        double angle = 2.0 * M_PI * i / sides;
        glVertex3d(x + r * cos(angle), y + r * sin(angle), z + h);
    }
    glEnd();

    // 绘制侧面
    glBegin(GL_QUAD_STRIP);
    for (int i = 0; i <= sides; ++i)
    {
        double angle = 2.0 * M_PI * i / sides;
        glVertex3d(x + r * cos(angle), y + r * sin(angle), z);
        glVertex3d(x + r * cos(angle), y + r * sin(angle), z + h);
    }
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // 恢复填充模式
}

void drawSphere(double x, double y, double z, double r, const GLfloat *color, bool drawWireframe)
{
    const int slices = 30;
    const int stacks = 30;

    GLfloat material_specular[] = {5e-7f, 5e-7f, 5e-7f, 1.0f}; // 镜面反射颜色
    // GLfloat material_specular[] = {0.1f, 0.1f, 0.1f, 1.0f};        // 镜面反射颜色
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0f);

    glColor3fv(color);

    if (drawWireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    for (int i = 0; i < slices; ++i)
    {
        double theta1 = (double)i / slices * 2.0 * M_PI;
        double theta2 = (double)(i + 1) / slices * 2.0 * M_PI;

        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= stacks; ++j)
        {
            double phi = (double)j / stacks * M_PI;

            double x1 = x + r * sin(phi) * cos(theta1);
            double y1 = y + r * sin(phi) * sin(theta1);
            double z1 = z + r * cos(phi);

            double x2 = x + r * sin(phi) * cos(theta2);
            double y2 = y + r * sin(phi) * sin(theta2);
            double z2 = z + r * cos(phi);

            glVertex3d(x1, y1, z1);
            glVertex3d(x2, y2, z2);
        }
        glEnd();
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // 恢复填充模式
}

void drawCube(double x, double y, double z, double lenx, double leny, double lenz, const GLfloat *color, bool drawWireframe)
{
    if (drawWireframe)
    {
        glColor3f(0.8f, 0.8f, 0.8f); // 灰色
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        glColor3fv(color);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // 前面
    glBegin(GL_QUADS);
    glVertex3d(x, y, z);
    glVertex3d(x + lenx, y, z);
    glVertex3d(x + lenx, y + leny, z);
    glVertex3d(x, y + leny, z);
    glEnd();

    // 后面
    glBegin(GL_QUADS);
    glVertex3d(x, y, z + lenz);
    glVertex3d(x + lenx, y, z + lenz);
    glVertex3d(x + lenx, y + leny, z + lenz);
    glVertex3d(x, y + leny, z + lenz);
    glEnd();

    // 左侧
    glBegin(GL_QUADS);
    glVertex3d(x, y, z);
    glVertex3d(x, y, z + lenz);
    glVertex3d(x, y + leny, z + lenz);
    glVertex3d(x, y + leny, z);
    glEnd();

    // 右侧
    glBegin(GL_QUADS);
    glVertex3d(x + lenx, y, z);
    glVertex3d(x + lenx, y, z + lenz);
    glVertex3d(x + lenx, y + leny, z + lenz);
    glVertex3d(x + lenx, y + leny, z);
    glEnd();

    // 顶面
    glBegin(GL_QUADS);
    glVertex3d(x, y + leny, z);
    glVertex3d(x + lenx, y + leny, z);
    glVertex3d(x + lenx, y + leny, z + lenz);
    glVertex3d(x, y + leny, z + lenz);
    glEnd();

    // 底面
    glBegin(GL_QUADS);
    glVertex3d(x, y, z);
    glVertex3d(x + lenx, y, z);
    glVertex3d(x + lenx, y, z + lenz);
    glVertex3d(x, y, z + lenz);
    glEnd();

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // 恢复填充模式
}
void drawPoints(const std::vector<Eigen::Vector3d> &points)
{
    glPointSize(3.0); // 设置点的大小

    glBegin(GL_POINTS);          // 开始绘制点集
    glColor3f(1.0f, 0.0f, 0.0f); // 设置点的颜色为红色

    for (const auto &point : points)
    {
        glVertex3d(point.x(), point.y(), point.z()); // 绘制每个点
    }

    glEnd(); // 结束绘制点集
}
void generateBoxSurfacePoints(double lx, double ly, double lz, double c, std::vector<Eigen::Vector3d> &points)
{
    points.clear(); // 清空输出的点集
    int nx = lx / c + 1;
    int ny = ly / c + 1;
    int nz = lz / c + 1;
    points.reserve(2 * (nx * ny + nz * ny + nx * nz));

    for (double y = -ly / 2.0; y <= ly / 2.0; y += c)
        for (double z = -lz / 2.0; z <= lz / 2.0; z += c)
        {
            points.push_back(Eigen::Vector3d(lx / 2.0, y, z));
            points.push_back(Eigen::Vector3d(-lx / 2.0, y, z));
        }
    for (double x = -lx / 2.0; x <= lx / 2.0; x += c)
        for (double z = -lz / 2.0; z <= lz / 2.0; z += c)
        {
            points.push_back(Eigen::Vector3d(x, ly / 2.0, z));
            points.push_back(Eigen::Vector3d(x, -ly / 2.0, z));
        }
    for (double x = -lx / 2.0; x <= lx / 2.0; x += c)
        for (double y = -ly / 2.0; y <= ly / 2.0; y += c)
        {
            points.push_back(Eigen::Vector3d(x, y, lz / 2.0));
            points.push_back(Eigen::Vector3d(x, y, -lz / 2.0));
        }
}

void drawCylinder(const CylinderSpace &space, const GLfloat *color, bool drawWireframe)
{
    drawCylinder(space.x, space.y, space.z, space.r, space.h, color, drawWireframe);
}
void drawSphere(const SphereSpace &space, const GLfloat *color, bool drawWireframe)
{
    drawSphere(space.x, space.y, space.z, space.r, color, drawWireframe);
}
void drawCube(const BoxSpace &space, const GLfloat *color, bool drawWireframe)
{
    drawCube(space.x, space.y, space.z, space.lenx, space.leny, space.lenz, color, drawWireframe);
}

// Function to draw mesh with normals, colors, and transformation
void DrawMesh(const aiScene *scene, bool wireframe, const GLfloat *color, const Eigen::Matrix4d &T)
{
    if (wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    else
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Set material color
    // GLfloat material_ambient_diffuse[] = {0.6f, 0.2f, 0.4f, 1.0f}; // 环境和漫射颜色
    GLfloat material_specular[] = {5e-7f, 5e-7f, 5e-7f, 1.0f}; // 镜面反射颜色
    // GLfloat material_specular[] = {0.1f, 0.1f, 0.1f, 1.0f};        // 镜面反射颜色
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
    glMaterialfv(GL_FRONT, GL_SPECULAR, material_specular);
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0f);
    // glColor3fv(color);
    glBegin(GL_TRIANGLES);
    for (unsigned int k = 0; k < scene->mNumMeshes; k++)
    {
        const aiMesh *mesh = scene->mMeshes[k];
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            const aiFace &face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
            {
                int index = face.mIndices[j];
                if (mesh->HasNormals())
                {
                    const aiVector3D *normal = &(mesh->mNormals[index]);
                    Eigen::Vector4d vertex(normal->x, normal->y, normal->z, 1.0);
                    Eigen::Vector4d transformed_vertex = T * vertex;
                    glNormal3f(transformed_vertex.x(), transformed_vertex.y(), transformed_vertex.z());
                }
                const aiVector3D *pos = &(mesh->mVertices[index]);

                // Apply transformation
                Eigen::Vector4d vertex(pos->x, pos->y, pos->z, 1.0);
                Eigen::Vector4d transformed_vertex = T * vertex;

                glVertex3f(transformed_vertex.x(), transformed_vertex.y(), transformed_vertex.z());
            }
        }
    }
    glEnd();

    // Reset to default fill mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

// void drawLine(const status &p1, const status &p2, const GLfloat *color)
// {
//     glColor3fv(color);
//     glBegin(GL_LINES);
//     glVertex3f(p1.x, p1.y, p1.z);
//     glVertex3f(p2.x, p2.y, p2.z);
//     glEnd();
// }
// void DrawPath(const std::list<status> &path, const GLfloat *color)
// {
//     glColor3fv(color);
//     glLineWidth(6.0f);
//     glBegin(GL_LINES);
//     status p1 = path.front();
//     for (const status &p2 : path)
//     {
//         glVertex3f(p1.x, p1.y, p1.z);
//         glVertex3f(p2.x, p2.y, p2.z);
//         p1 = p2;
//     }
//     glEnd();
//     glLineWidth(2.0f);
// }
