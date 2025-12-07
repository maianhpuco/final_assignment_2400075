#include "gpu_raytracer.cuh"
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
{                                                                              \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA error at line %d: %s (%d)\n",                             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define GPU_INFINITY 1e20f
#define GPU_PI 3.14159265358979323846f

__device__ float gpuRandomFloat(curandState* state) {
    return curand_uniform(state);
}

__device__ bool gpuHitSphere(const CudaSphere& sphere, const CudaRay& ray,
                              float tMin, float tMax, 
                              CudaVec3& hitPoint, CudaVec3& hitNormal, 
                              CudaMaterial& hitMaterial, float& hitT) {
    CudaVec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) return false;

    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    
    if (root < tMin || root > tMax) {
        root = (-half_b + sqrtd) / a;
        if (root < tMin || root > tMax) {
            return false;
        }
    }

    hitT = root;
    hitPoint = ray.at(hitT);
    CudaVec3 outward_normal = (hitPoint - sphere.center) / sphere.radius;
    
    // Determine front face
    bool front_face = dot(ray.direction, outward_normal) < 0;
    hitNormal = front_face ? outward_normal : outward_normal * (-1.0f);
    hitMaterial = sphere.material;
    
    return true;
}

__device__ bool gpuHitWorld(const CudaSphere* spheres, int numSpheres,
                             const CudaRay& ray, float tMin, float tMax,
                             CudaVec3& hitPoint, CudaVec3& hitNormal,
                             CudaMaterial& hitMaterial, float& hitT) {
    bool hitAnything = false;
    float closest = tMax;
    
    CudaVec3 tempPoint, tempNormal;
    CudaMaterial tempMaterial;
    float tempT;

    for (int i = 0; i < numSpheres; i++) {
        if (gpuHitSphere(spheres[i], ray, tMin, closest, tempPoint, tempNormal, tempMaterial, tempT)) {
            hitAnything = true;
            closest = tempT;
            hitPoint = tempPoint;
            hitNormal = tempNormal;
            hitMaterial = tempMaterial;
            hitT = tempT;
        }
    }

    return hitAnything;
}

__device__ CudaVec3 gpuSampleHemisphere(const CudaVec3& normal, curandState* state) {
    float x0 = gpuRandomFloat(state);
    float x1 = gpuRandomFloat(state);
    float phi = 2.0f * GPU_PI * x1;
    float theta = acosf(sqrtf(1.0f - x0));
    
    float sinTheta = sinf(theta);
    float cosTheta = cosf(theta);
    float sinPhi = sinf(phi);
    float cosPhi = cosf(phi);

    CudaVec3 w = normal.normalized();
    CudaVec3 a = (fabsf(w.x) > 0.9f) ? CudaVec3(0, 1, 0) : CudaVec3(1, 0, 0);
    CudaVec3 v = cross(w, a).normalized();
    CudaVec3 u = cross(w, v);

    CudaVec3 dir = u * (sinTheta * cosPhi) + v * (sinTheta * sinPhi) + w * cosTheta;
    return dir.normalized();
}

__device__ CudaVec3 gpuRayColor(const CudaRay& ray, const CudaSphere* spheres,
                                 int numSpheres, int maxBounces, curandState* state) {
    CudaRay currentRay = ray;
    CudaVec3 throughput(1.0f, 1.0f, 1.0f);
    CudaVec3 accumulatedColor(0.0f, 0.0f, 0.0f);
    
    for (int bounce = 0; bounce < maxBounces; bounce++) {
        CudaVec3 hitPoint, hitNormal;
        CudaMaterial hitMaterial;
        float hitT;
        
        if (gpuHitWorld(spheres, numSpheres, currentRay, 0.001f, GPU_INFINITY,
                        hitPoint, hitNormal, hitMaterial, hitT)) {
            accumulatedColor = accumulatedColor + throughput * hitMaterial.emissionColor * hitMaterial.emission;
            
            CudaVec3 newDir = gpuSampleHemisphere(hitNormal, state);
            throughput = throughput * hitMaterial.albedo * hitMaterial.scattering;
            currentRay = CudaRay(hitPoint, newDir);
        } else {
            CudaVec3 unitDir = currentRay.direction.normalized();
            float t = 0.5f * (unitDir.y + 1.0f);
            CudaVec3 skyColor = CudaVec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + 
                                CudaVec3(0.5f, 0.7f, 1.0f) * t;
            accumulatedColor = accumulatedColor + throughput * skyColor;
            return accumulatedColor;
        }
    }
    
    return accumulatedColor;
}

__global__ void gpuInitRandStatesKernel(curandState* states, int width, int height, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    curand_init(seed + idx, 0, 0, &states[idx]);
}

__global__ void gpuPathTracingKernel(
    float* outputBuffer,
    float* accumBuffer,
    const CudaSphere* spheres,
    int numSpheres,
    CudaCameraParams camera,
    curandState* randStates,
    int maxBounces,
    int sampleCount
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= camera.image_width || y >= camera.image_height) return;
    
    int pixelIdx = y * camera.image_width + x;
    curandState* localState = &randStates[pixelIdx];
    
    // Generate ray with random offset for anti-aliasing
    float offsetX = gpuRandomFloat(localState) - 0.5f;
    float offsetY = gpuRandomFloat(localState) - 0.5f;
    
    CudaVec3 pixelCenter = camera.pixel00_loc 
        + camera.pixel_delta_u * (x + offsetX)
        + camera.pixel_delta_v * (y + offsetY);
    
    CudaVec3 rayDir = pixelCenter - camera.origin;
    CudaRay ray(camera.origin, rayDir);
    
    // Trace ray (using global memory for spheres)
    CudaVec3 color = gpuRayColor(ray, spheres, numSpheres, maxBounces, localState);
    
    // Accumulate samples
    int bufferIdx = pixelIdx * 4;
    
    if (sampleCount == 1) {
        accumBuffer[bufferIdx + 0] = color.x;
        accumBuffer[bufferIdx + 1] = color.y;
        accumBuffer[bufferIdx + 2] = color.z;
        accumBuffer[bufferIdx + 3] = 1.0f;
    } else {
        accumBuffer[bufferIdx + 0] += color.x;
        accumBuffer[bufferIdx + 1] += color.y;
        accumBuffer[bufferIdx + 2] += color.z;
    }
    
    // Output averaged color
    float invSamples = 1.0f / (float)sampleCount;
    outputBuffer[bufferIdx + 0] = accumBuffer[bufferIdx + 0] * invSamples;
    outputBuffer[bufferIdx + 1] = accumBuffer[bufferIdx + 1] * invSamples;
    outputBuffer[bufferIdx + 2] = accumBuffer[bufferIdx + 2] * invSamples;
    outputBuffer[bufferIdx + 3] = 1.0f;
}

void initGpuRandStates(void* states, int width, int height, unsigned long seed) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gpuInitRandStatesKernel<<<gridSize, blockSize>>>((curandState*)states, width, height, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

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
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((camera.image_width + blockSize.x - 1) / blockSize.x,
                  (camera.image_height + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(start);
    
    gpuPathTracingKernel<<<gridSize, blockSize>>>(
        outputBuffer,
        accumBuffer,
        spheres,
        numSpheres,
        camera,
        (curandState*)randStates,
        maxBounces,
        sampleCount
    );
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&kernelTimeMs, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    CHECK_CUDA(cudaGetLastError());
}

GpuRaytracer::GpuRaytracer()
    : d_spheres(nullptr)
    , d_accumBuffer(nullptr)
    , d_outputBuffer(nullptr)
    , d_randStates(nullptr)
    , m_numSpheres(0)
    , m_imageWidth(0)
    , m_imageHeight(0)
    , m_numBounces(5)
    , m_accumulate(true)
    , m_sampleCount(0)
    , m_kernelTimeMs(0)
    , m_memoryTransferTimeMs(0)
    , m_initialized(false)
{
}

GpuRaytracer::~GpuRaytracer()
{
    cleanup();
}

void GpuRaytracer::init(
    const CudaCameraParams& camera,
    const CudaSphere* spheres,
    int numSpheres,
    int imageWidth,
    int imageHeight)
{
    cleanup();
    
    m_camera = camera;
    m_numSpheres = numSpheres;
    m_imageWidth = imageWidth;
    m_imageHeight = imageHeight;
    m_sampleCount = 0;
    
    int numPixels = imageWidth * imageHeight;
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_spheres, numSpheres * sizeof(CudaSphere)));
    CHECK_CUDA(cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(CudaSphere), cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMalloc(&d_accumBuffer, numPixels * 4 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_accumBuffer, 0, numPixels * 4 * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&d_outputBuffer, numPixels * 4 * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&d_randStates, numPixels * sizeof(curandState)));
    
    initGpuRandStates(d_randStates, imageWidth, imageHeight, 1234);
    
    m_initialized = true;
    
    printf("GPU Ray Tracer (Global Memory) initialized:\n");
    printf("  Image: %d x %d\n", imageWidth, imageHeight);
    printf("  Spheres: %d\n", numSpheres);
}

void GpuRaytracer::setNumBounces(int numBounces)
{
    m_numBounces = (numBounces > 0) ? numBounces : 1;
}

void GpuRaytracer::setAccumulation(bool enable)
{
    if (m_accumulate && !enable)
    {
        resetAccumulation();
    }
    m_accumulate = enable;
}

void GpuRaytracer::resetAccumulation()
{
    m_sampleCount = 0;
    if (d_accumBuffer && m_initialized)
    {
        CHECK_CUDA(cudaMemset(d_accumBuffer, 0, m_imageWidth * m_imageHeight * 4 * sizeof(float)));
    }
}

void GpuRaytracer::render(float* outputImage)
{
    if (!m_initialized || !outputImage) return;
    
    if (!m_accumulate)
    {
        m_sampleCount = 0;
        CHECK_CUDA(cudaMemset(d_accumBuffer, 0, m_imageWidth * m_imageHeight * 4 * sizeof(float)));
    }
    
    m_sampleCount++;
    
    // Launch kernel
    launchGpuPathTracingKernel(
        d_outputBuffer,
        d_accumBuffer,
        d_spheres,
        m_numSpheres,
        m_camera,
        d_randStates,
        m_numBounces,
        m_sampleCount,
        m_kernelTimeMs
    );
    
    // Copy result back to host
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    CHECK_CUDA(cudaMemcpy(outputImage, d_outputBuffer,
                          m_imageWidth * m_imageHeight * 4 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&m_memoryTransferTimeMs, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

GpuRenderStats GpuRaytracer::getStats() const
{
    GpuRenderStats stats;
    stats.samplesRendered = m_sampleCount;
    stats.kernelTimeMs = m_kernelTimeMs;
    stats.memoryTransferTimeMs = m_memoryTransferTimeMs;
    stats.totalTimeMs = m_kernelTimeMs + m_memoryTransferTimeMs;
    return stats;
}

void GpuRaytracer::cleanup()
{
    if (d_spheres) { cudaFree(d_spheres); d_spheres = nullptr; }
    if (d_accumBuffer) { cudaFree(d_accumBuffer); d_accumBuffer = nullptr; }
    if (d_outputBuffer) { cudaFree(d_outputBuffer); d_outputBuffer = nullptr; }
    if (d_randStates) { cudaFree(d_randStates); d_randStates = nullptr; }
    
    m_sampleCount = 0;
    m_initialized = false;
}
