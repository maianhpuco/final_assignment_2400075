#include "cpu_raytracer.h"

#include <chrono>
#include <future>
#include <cstdio>

#define CPU_INFINITY 1e20f
#define CPU_PI 3.14159265358979323846f

CpuRaytracer::CpuRaytracer()
    : m_numSpheres(0)
    , m_imageWidth(0)
    , m_imageHeight(0)
    , m_numBounces(5)
    , m_numThreads(4)
    , m_accumulate(true)
    , m_sampleCount(0)
    , m_renderTimeMs(0.0f)
    , m_initialized(false)
{
}

CpuRaytracer::~CpuRaytracer()
{
    cleanup();
}

void CpuRaytracer::init(
    const CpuCameraParams& camera,
    const CpuSphere* spheres,
    int numSpheres,
    int imageWidth,
    int imageHeight)
{
    m_camera = camera;
    m_imageWidth = imageWidth;
    m_imageHeight = imageHeight;
    m_numSpheres = numSpheres;

    // Copy spheres
    m_spheres.resize(numSpheres);
    for (int i = 0; i < numSpheres; i++) {
        m_spheres[i] = spheres[i];
    }

    // Allocate accumulation buffer (RGBA per pixel)
    m_accumBuffer.resize(imageWidth * imageHeight * 4, 0.0f);
    m_sampleCount = 0;

    // Initialize random generators for each potential thread
    m_randGenerators.resize(256);  // Support up to 256 threads
    std::random_device rd;
    for (int i = 0; i < 256; i++) {
        m_randGenerators[i].seed(rd() + i);
    }

    m_initialized = true;
    printf("CPU Raytracer initialized: %d x %d, %d spheres\n", imageWidth, imageHeight, numSpheres);
}

void CpuRaytracer::setNumBounces(int numBounces)
{
    m_numBounces = (numBounces > 0) ? numBounces : 1;
}

void CpuRaytracer::setNumThreads(int numThreads)
{
    m_numThreads = (numThreads > 0) ? numThreads : 1;
    if (m_numThreads > 16) m_numThreads = 16;
}

void CpuRaytracer::setAccumulation(bool enable)
{
    if (m_accumulate && !enable) {
        resetAccumulation();
    }
    m_accumulate = enable;
}

void CpuRaytracer::resetAccumulation()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), 0.0f);
    m_sampleCount = 0;
}

void CpuRaytracer::cleanup()
{
    m_spheres.clear();
    m_accumBuffer.clear();
    m_randGenerators.clear();
    m_initialized = false;
}

CpuRenderStats CpuRaytracer::getStats() const
{
    CpuRenderStats stats;
    stats.samplesRendered = m_sampleCount;
    stats.renderTimeMs = m_renderTimeMs;
    stats.totalTimeMs = m_renderTimeMs;
    return stats;
}

CpuVec3 CpuRaytracer::sampleHemisphere(const CpuVec3& normal, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    float x0 = dist(rng);
    float x1 = dist(rng);
    float phi = 2.0f * CPU_PI * x1;
    float theta = std::acos(std::sqrt(1.0f - x0));
    
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);

    CpuVec3 w = normal.normalized();
    CpuVec3 a = (std::fabs(w.x) > 0.9f) ? CpuVec3(0, 1, 0) : CpuVec3(1, 0, 0);
    CpuVec3 v = cross(w, a).normalized();
    CpuVec3 u = cross(w, v);

    CpuVec3 dir = u * (sinTheta * cosPhi) + v * (sinTheta * sinPhi) + w * cosTheta;
    return dir.normalized();
}

bool CpuRaytracer::hitSphere(const CpuSphere& sphere, const CpuRay& ray,
                              float tMin, float tMax,
                              CpuVec3& hitPoint, CpuVec3& hitNormal,
                              CpuMaterial& hitMaterial, float& hitT)
{
    CpuVec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0) return false;

    float sqrtd = std::sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    
    if (root < tMin || root > tMax) {
        root = (-half_b + sqrtd) / a;
        if (root < tMin || root > tMax) {
            return false;
        }
    }

    hitT = root;
    hitPoint = ray.at(hitT);
    CpuVec3 outward_normal = (hitPoint - sphere.center) / sphere.radius;
    
    bool front_face = dot(ray.direction, outward_normal) < 0;
    hitNormal = front_face ? outward_normal : outward_normal * (-1.0f);
    hitMaterial = sphere.material;
    
    return true;
}

bool CpuRaytracer::hitWorld(const CpuRay& ray, float tMin, float tMax,
                             CpuVec3& hitPoint, CpuVec3& hitNormal,
                             CpuMaterial& hitMaterial, float& hitT)
{
    bool hitAnything = false;
    float closest = tMax;
    
    CpuVec3 tempPoint, tempNormal;
    CpuMaterial tempMaterial;
    float tempT;

    for (int i = 0; i < m_numSpheres; i++) {
        if (hitSphere(m_spheres[i], ray, tMin, closest, tempPoint, tempNormal, tempMaterial, tempT)) {
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

CpuVec3 CpuRaytracer::rayColor(const CpuRay& ray, int depth, std::mt19937& rng)
{
    CpuRay currentRay = ray;
    CpuVec3 throughput(1.0f, 1.0f, 1.0f);
    CpuVec3 accumulatedColor(0.0f, 0.0f, 0.0f);
    
    for (int bounce = 0; bounce < depth; bounce++) {
        CpuVec3 hitPoint, hitNormal;
        CpuMaterial hitMaterial;
        float hitT;
        
        if (hitWorld(currentRay, 0.001f, CPU_INFINITY, hitPoint, hitNormal, hitMaterial, hitT)) {
            accumulatedColor = accumulatedColor + throughput * hitMaterial.emissionColor * hitMaterial.emission;
            
            CpuVec3 newDir = sampleHemisphere(hitNormal, rng);
            throughput = throughput * hitMaterial.albedo * hitMaterial.scattering;
            currentRay = CpuRay(hitPoint, newDir);
        } else {
            CpuVec3 unitDir = currentRay.direction.normalized();
            float t = 0.5f * (unitDir.y + 1.0f);
            CpuVec3 skyColor = CpuVec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + 
                               CpuVec3(0.5f, 0.7f, 1.0f) * t;
            accumulatedColor = accumulatedColor + throughput * skyColor;
            return accumulatedColor;
        }
    }
    
    return accumulatedColor;
}

void CpuRaytracer::renderTile(int startX, int startY, int endX, int endY, int threadIdx)
{
    std::mt19937& rng = m_randGenerators[threadIdx % m_randGenerators.size()];
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int y = startY; y < endY; y++) {
        for (int x = startX; x < endX; x++) {
            // Generate ray with jitter for anti-aliasing
            float u = dist(rng);
            float v = dist(rng);
            
            CpuVec3 pixelCenter = m_camera.pixel00_loc + 
                                  ((float)x + u) * m_camera.pixel_delta_u + 
                                  ((float)y + v) * m_camera.pixel_delta_v;
            CpuVec3 rayDir = (pixelCenter - m_camera.origin).normalized();
            CpuRay ray(m_camera.origin, rayDir);
            
            // Trace ray
            CpuVec3 pixelColor = rayColor(ray, m_numBounces, rng);
            
            // Accumulate
            int idx = (y * m_imageWidth + x) * 4;
            {
                // std::lock_guard<std::mutex> lock(m_mutex);
                m_accumBuffer[idx + 0] += pixelColor.x;
                m_accumBuffer[idx + 1] += pixelColor.y;
                m_accumBuffer[idx + 2] += pixelColor.z;
                m_accumBuffer[idx + 3] += 1.0f;
            }
        }
    }
}

void CpuRaytracer::render(float* outputImage)
{
    if (!m_initialized || !outputImage) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Reset if not accumulating
    if (!m_accumulate) {
        std::fill(m_accumBuffer.begin(), m_accumBuffer.end(), 0.0f);
        m_sampleCount = 0;
    }

    // Calculate tile size
    int tileWidth = (m_imageWidth + m_numThreads - 1) / m_numThreads;
    int tileHeight = (m_imageHeight + m_numThreads - 1) / m_numThreads;

    // Launch threads
    std::vector<std::future<void>> futures;
    int threadIdx = 0;
    
    for (int ty = 0; ty < m_numThreads; ty++) {
        for (int tx = 0; tx < m_numThreads; tx++) {
            int startX = tx * tileWidth;
            int startY = ty * tileHeight;
            int endX = std::min(startX + tileWidth, m_imageWidth);
            int endY = std::min(startY + tileHeight, m_imageHeight);
            
            if (startX < m_imageWidth && startY < m_imageHeight) {
                futures.push_back(std::async(std::launch::async, 
                    &CpuRaytracer::renderTile, this, startX, startY, endX, endY, threadIdx++));
            }
        }
    }

    // Wait for all threads
    for (auto& f : futures) {
        f.wait();
    }

    m_sampleCount++;

    // Output averaged result
    float invSamples = 1.0f / (float)m_sampleCount;
    for (int i = 0; i < m_imageWidth * m_imageHeight; i++) {
        int idx = i * 4;
        outputImage[idx + 0] = m_accumBuffer[idx + 0] * invSamples;
        outputImage[idx + 1] = m_accumBuffer[idx + 1] * invSamples;
        outputImage[idx + 2] = m_accumBuffer[idx + 2] * invSamples;
        outputImage[idx + 3] = 1.0f;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = endTime - startTime;
    m_renderTimeMs = duration.count();
}
