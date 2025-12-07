#pragma once

#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <random>

struct CpuVec3 {
    float x, y, z;

    CpuVec3() : x(0), y(0), z(0) {}
    CpuVec3(float val) : x(val), y(val), z(val) {}
    CpuVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    CpuVec3 operator+(const CpuVec3& v) const {
        return CpuVec3(x + v.x, y + v.y, z + v.z);
    }

    CpuVec3 operator-(const CpuVec3& v) const {
        return CpuVec3(x - v.x, y - v.y, z - v.z);
    }

    CpuVec3 operator*(float t) const {
        return CpuVec3(x * t, y * t, z * t);
    }

    CpuVec3 operator*(const CpuVec3& v) const {
        return CpuVec3(x * v.x, y * v.y, z * v.z);
    }

    CpuVec3 operator/(float t) const {
        return CpuVec3(x / t, y / t, z / t);
    }

    CpuVec3& operator+=(const CpuVec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    float length_squared() const {
        return x * x + y * y + z * z;
    }

    CpuVec3 normalized() const {
        float len = length();
        return len > 0 ? *this / len : CpuVec3(0);
    }
};

inline CpuVec3 operator*(float t, const CpuVec3& v) {
    return v * t;
}

inline float dot(const CpuVec3& a, const CpuVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline CpuVec3 cross(const CpuVec3& a, const CpuVec3& b) {
    return CpuVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

struct CpuRay {
    CpuVec3 origin;
    CpuVec3 direction;

    CpuRay() {}
    CpuRay(const CpuVec3& o, const CpuVec3& d) : origin(o), direction(d) {}

    CpuVec3 at(float t) const {
        return origin + direction * t;
    }
};

struct CpuMaterial {
    CpuVec3 albedo;
    CpuVec3 emissionColor;
    float emission;
    float scattering;

    CpuMaterial() : albedo(1, 1, 1), emissionColor(1, 1, 1), emission(0.5f), scattering(0.5f) {}
    CpuMaterial(const CpuVec3& c, float e = 0.5f, float s = 0.5f)
        : albedo(c), emissionColor(1, 1, 1), emission(e), scattering(s) {}
    CpuMaterial(const CpuVec3& c, float e, float s, const CpuVec3& ec)
        : albedo(c), emissionColor(ec), emission(e), scattering(s) {}
};

struct CpuSphere {
    CpuVec3 center;
    float radius;
    CpuMaterial material;

    CpuSphere() : center(0), radius(1) {}
    CpuSphere(const CpuVec3& c, float r, const CpuMaterial& m)
        : center(c), radius(r), material(m) {}
};

struct CpuCameraParams {
    CpuVec3 origin;
    CpuVec3 pixel00_loc;
    CpuVec3 pixel_delta_u;
    CpuVec3 pixel_delta_v;
    int image_width;
    int image_height;
};

struct CpuRenderStats {
    int samplesRendered;
    float renderTimeMs;
    float totalTimeMs;
};

class CpuRaytracer
{
public:
    CpuRaytracer();
    ~CpuRaytracer();

    // Initialize with scene data
    void init(
        const CpuCameraParams& camera,
        const CpuSphere* spheres,
        int numSpheres,
        int imageWidth,
        int imageHeight
    );

    // Set parameters
    void setNumBounces(int numBounces);
    void setNumThreads(int numThreads);
    void setAccumulation(bool enable);

    // Render one frame
    void render(float* outputImage);

    // Get statistics
    CpuRenderStats getStats() const;
    int getSampleCount() const { return m_sampleCount; }

    // Reset accumulation
    void resetAccumulation();

    // Cleanup
    void cleanup();

private:
    // Scene data
    std::vector<CpuSphere> m_spheres;
    std::vector<float> m_accumBuffer;

    // Camera
    CpuCameraParams m_camera;

    // Parameters
    int m_numSpheres;
    int m_imageWidth;
    int m_imageHeight;
    int m_numBounces;
    int m_numThreads;
    bool m_accumulate;

    // Statistics
    int m_sampleCount;
    float m_renderTimeMs;

    bool m_initialized;

    // Thread-local random generators
    std::vector<std::mt19937> m_randGenerators;
    std::mutex m_mutex;

    // Internal functions
    void renderTile(int startX, int startY, int endX, int endY, int threadIdx);
    CpuVec3 rayColor(const CpuRay& ray, int depth, std::mt19937& rng);
    CpuVec3 sampleHemisphere(const CpuVec3& normal, std::mt19937& rng);
    bool hitSphere(const CpuSphere& sphere, const CpuRay& ray, float tMin, float tMax,
                   CpuVec3& hitPoint, CpuVec3& hitNormal, CpuMaterial& hitMaterial, float& hitT);
    bool hitWorld(const CpuRay& ray, float tMin, float tMax,
                  CpuVec3& hitPoint, CpuVec3& hitNormal, CpuMaterial& hitMaterial, float& hitT);
};
