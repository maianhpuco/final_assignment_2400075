#pragma once

#include <cuda_runtime.h>

struct CudaVec3 {
    float x, y, z;

    __host__ __device__ CudaVec3() : x(0), y(0), z(0) {}
    __host__ __device__ CudaVec3(float val) : x(val), y(val), z(val) {}
    __host__ __device__ CudaVec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ CudaVec3 operator+(const CudaVec3& v) const {
        return CudaVec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ CudaVec3 operator-(const CudaVec3& v) const {
        return CudaVec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ CudaVec3 operator*(float t) const {
        return CudaVec3(x * t, y * t, z * t);
    }

    __host__ __device__ CudaVec3 operator*(const CudaVec3& v) const {
        return CudaVec3(x * v.x, y * v.y, z * v.z);
    }

    __host__ __device__ CudaVec3 operator/(float t) const {
        return CudaVec3(x / t, y / t, z / t);
    }

    __host__ __device__ CudaVec3& operator+=(const CudaVec3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ float length_squared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ CudaVec3 normalized() const {
        float len = length();
        return len > 0 ? *this / len : CudaVec3(0);
    }
};

__host__ __device__ inline CudaVec3 operator*(float t, const CudaVec3& v) {
    return v * t;
}

__host__ __device__ inline float dot(const CudaVec3& a, const CudaVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline CudaVec3 cross(const CudaVec3& a, const CudaVec3& b) {
    return CudaVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

struct CudaRay {
    CudaVec3 origin;
    CudaVec3 direction;

    __host__ __device__ CudaRay() {}
    __host__ __device__ CudaRay(const CudaVec3& o, const CudaVec3& d) : origin(o), direction(d) {}

    __host__ __device__ CudaVec3 at(float t) const {
        return origin + direction * t;
    }
};

struct CudaMaterial {
    CudaVec3 albedo;
    CudaVec3 emissionColor;
    float emission;
    float scattering;

    __host__ __device__ CudaMaterial() : albedo(1, 1, 1), emissionColor(1, 1, 1), emission(0.5f), scattering(0.5f) {}
    __host__ __device__ CudaMaterial(const CudaVec3& c, float e = 0.5f, float s = 0.5f)
        : albedo(c), emissionColor(1, 1, 1), emission(e), scattering(s) {}
    __host__ __device__ CudaMaterial(const CudaVec3& c, float e, float s, const CudaVec3& ec)
        : albedo(c), emissionColor(ec), emission(e), scattering(s) {}
};

struct CudaSphere {
    CudaVec3 center;
    float radius;
    CudaMaterial material;

    __host__ __device__ CudaSphere() : center(0), radius(1) {}
    __host__ __device__ CudaSphere(const CudaVec3& c, float r, const CudaMaterial& m)
        : center(c), radius(r), material(m) {}
};

struct CudaCameraParams {
    CudaVec3 origin;
    CudaVec3 pixel00_loc;
    CudaVec3 pixel_delta_u;
    CudaVec3 pixel_delta_v;
    int image_width;
    int image_height;
};

struct GpuRenderStats {
    int samplesRendered;
    float kernelTimeMs;
    float memoryTransferTimeMs;
    float totalTimeMs;
};

class GpuRaytracer
{
public:
    GpuRaytracer();
    ~GpuRaytracer();

    // Initialize with scene data
    void init(
        const CudaCameraParams& camera,
        const CudaSphere* spheres,
        int numSpheres,
        int imageWidth,
        int imageHeight
    );

    // Set parameters
    void setNumBounces(int numBounces);
    void setAccumulation(bool enable);

    // Render one frame
    void render(float* outputImage);

    // Get statistics
    GpuRenderStats getStats() const;

    // Reset accumulation
    void resetAccumulation();

    // Cleanup
    void cleanup();

private:
    // Device pointers
    CudaSphere* d_spheres;
    float* d_accumBuffer;
    float* d_outputBuffer;
    void* d_randStates;  // curandState*

    // Host copies
    CudaCameraParams m_camera;
    int m_numSpheres;
    int m_imageWidth;
    int m_imageHeight;
    int m_numBounces;
    bool m_accumulate;

    // Statistics
    int m_sampleCount;
    float m_kernelTimeMs;
    float m_memoryTransferTimeMs;

    bool m_initialized;
};

void launchGpuPathTracingKernel(
    float* outputBuffer,
    float* accumBuffer,
    const CudaSphere* spheres,
    int numSpheres,
    CudaCameraParams camera,
    void* randStates,
    int maxBounces,
    int sampleCount,
    float& kernelTimeMs
);

void initGpuRandStates(void* states, int width, int height, unsigned long seed);
