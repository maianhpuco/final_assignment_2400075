#pragma once

#include "gpu_raytracer.cuh"


// Maximum spheres that can fit in shared memory
#define MAX_SHARED_SPHERES 64

class GpuSharedRaytracer
{
public:
    GpuSharedRaytracer();
    ~GpuSharedRaytracer();

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

void launchGpuSharedPathTracingKernel(
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

void initGpuSharedRandStates(void* states, int width, int height, unsigned long seed);
