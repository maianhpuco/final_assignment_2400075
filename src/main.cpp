/*
	CUDA Path Tracing Ray Tracer
	
	This application demonstrates three rendering modes:
	1. CPU - Multi-threaded CPU ray tracing
	2. GPU (Global Memory) - CUDA ray tracing using global memory
	3. GPU (Shared Memory) - CUDA ray tracing with shared memory optimization
	
	Libraries used:
	- OpenGL for display
	- GLFW for window management
	- ImGui for UI
	- CUDA for GPU computation
*/

#include "helloworld.h"
#include "utility.h"
#include "vec3.h"

// Raytracer implementations
#include "cpu_raytracer.h"
#include "gpu_raytracer.cuh"
#include "gpu_shared_raytracer.cuh"

#include <iostream>
#include <vector>
#include <chrono>

// Global variables
int    resolution = 500;
float* output_image_ptr = nullptr;
float  frame_seconds = 0.0f;
int    num_of_samples_rendered = 1;
bool   is_accumulation = true;
int    num_of_threads = 4;
int    num_of_bounces = 5;
RenderModeUI render_mode = RenderModeUI::CPU;

// Performance stats
float  gpu_kernel_time_ms = 0.0f;
float  gpu_memory_time_ms = 0.0f;

// Helper function to create scene spheres for CPU
std::vector<CpuSphere> createCpuScene()
{
	std::vector<CpuSphere> spheres;
	
	const float PI = 3.14159265358979323846f;
	
	// Ground sphere
	spheres.push_back(CpuSphere(
		CpuVec3(0, -1000, 0), 1000,
		CpuMaterial(CpuVec3(0.6f, 0.6f, 0.6f), 0.0f, 1.0f)
	));
	
	const int N = 12;
	const float R = 2.55f;
	const float rObj = 0.62f;
	const float objectY = rObj;
	const float lightY = 6.0f;
	const float rLight = 0.5f;
	const float E = 30.0f;
	
	// Diffuse ring
	for (int i = 0; i < N; ++i)
	{
		float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(N);
		float x = R * std::cos(a);
		float z = R * std::sin(a);
		
		spheres.push_back(CpuSphere(
			CpuVec3(x, objectY, z), rObj,
			CpuMaterial(CpuVec3(0.8f, 0.8f, 0.8f), 0.0f, 1.0f)
		));
	}
	
	// Emissive ring
	for (int i = 0; i < N; ++i)
	{
		float a = 2.0f * PI * static_cast<float>(i) / static_cast<float>(N);
		float x = R * std::cos(a);
		float z = R * std::sin(a);
		
		CpuVec3 emissionColor(
			0.55f + 0.35f * std::cos(a + 0.0f),
			0.35f + 0.20f * std::cos(a + 2.094f),
			0.55f + 0.35f * std::cos(a + 4.188f)
		);
		
		spheres.push_back(CpuSphere(
			CpuVec3(x, lightY, z), rLight,
			CpuMaterial(CpuVec3(1.0f, 1.0f, 1.0f), E, 1.0f, emissionColor)
		));
	}
	
	return spheres;
}

std::vector<CudaSphere> cpuSphere2Gpu(const std::vector<CpuSphere>& cpuSpheres)
{
	std::vector<CudaSphere> cudaSpheres;
	cudaSpheres.reserve(cpuSpheres.size());
	
	for (const auto& cs : cpuSpheres)
	{
		cudaSpheres.push_back(CudaSphere(
			CudaVec3(cs.center.x, cs.center.y, cs.center.z),
			cs.radius,
			CudaMaterial(
				CudaVec3(cs.material.albedo.x, cs.material.albedo.y, cs.material.albedo.z),
				cs.material.emission,
				cs.material.scattering,
				CudaVec3(cs.material.emissionColor.x, cs.material.emissionColor.y, cs.material.emissionColor.z)
			)
		));
	}
	
	return cudaSpheres;
}

// Main function
int main(int argc, const char* argv[])
{
	// Initialize GLFW and create window
	if (!glfwInit())
		throw std::runtime_error("Failed to initialize GLFW");

	GLFWwindow* window = glfwCreateWindow(1920, 1080, "CUDA Path Tracer", nullptr, nullptr);
	if (window == nullptr)
		throw std::runtime_error("Failed to create GLFW window");

	glfwMakeContextCurrent(window);

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	// Initialize GLEW
	if (glewInit() != GLEW_OK)
		throw std::runtime_error("Failed to initialize GLEW");

	// Allocate output image
	output_image_ptr = new float[resolution * resolution * 4];

	// Setup camera parameters
	// Camera settings
	double aspect_ratio = 1.0;
	int image_width = resolution;
	double vfov = 35;
	vec3 lookfrom(0, 5, 14);
	vec3 lookat(0, 1.0, 0);
	vec3 vup(0, 1, 0);

	int image_height = (int)(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;
	
	double focal_length = (lookfrom - lookat).length();
	double theta = degrees_to_radians(vfov);
	double h = std::tan(theta / 2);
	double viewport_height = 2 * h * focal_length;
	double viewport_width = viewport_height * ((double)image_width / image_height);
	
	vec3 w = unit_vector(lookfrom - lookat);
	vec3 u = unit_vector(cross(vup, w));
	vec3 v = cross(w, u);
	
	vec3 viewport_u = viewport_width * u;
	vec3 viewport_v = viewport_height * (-v);
	vec3 pixel_delta_u = viewport_u / image_width;
	vec3 pixel_delta_v = viewport_v / image_height;
	vec3 viewport_upper_left = lookfrom - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
	vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

	// Setup CUDA camera parameters
	CudaCameraParams cudaCamera;
	cudaCamera.origin = CudaVec3((float)lookfrom.x(), (float)lookfrom.y(), (float)lookfrom.z());
	cudaCamera.pixel00_loc = CudaVec3((float)pixel00_loc.x(), (float)pixel00_loc.y(), (float)pixel00_loc.z());
	cudaCamera.pixel_delta_u = CudaVec3((float)pixel_delta_u.x(), (float)pixel_delta_u.y(), (float)pixel_delta_u.z());
	cudaCamera.pixel_delta_v = CudaVec3((float)pixel_delta_v.x(), (float)pixel_delta_v.y(), (float)pixel_delta_v.z());
	cudaCamera.image_width = resolution;
	cudaCamera.image_height = resolution;

	// Setup CPU camera parameters (same as CUDA)
	CpuCameraParams cpuCamera;
	cpuCamera.origin = CpuVec3((float)lookfrom.x(), (float)lookfrom.y(), (float)lookfrom.z());
	cpuCamera.pixel00_loc = CpuVec3((float)pixel00_loc.x(), (float)pixel00_loc.y(), (float)pixel00_loc.z());
	cpuCamera.pixel_delta_u = CpuVec3((float)pixel_delta_u.x(), (float)pixel_delta_u.y(), (float)pixel_delta_u.z());
	cpuCamera.pixel_delta_v = CpuVec3((float)pixel_delta_v.x(), (float)pixel_delta_v.y(), (float)pixel_delta_v.z());
	cpuCamera.image_width = resolution;
	cpuCamera.image_height = resolution;

	// Create scenes
	std::vector<CpuSphere> cpuSpheres = createCpuScene();
	std::vector<CudaSphere> cudaSpheres = cpuSphere2Gpu(cpuSpheres);

	// Initialize all three renderers
	CpuRaytracer cpuRenderer;
	cpuRenderer.init(cpuCamera, cpuSpheres.data(), (int)cpuSpheres.size(), resolution, resolution);
	cpuRenderer.setNumThreads(num_of_threads);
	cpuRenderer.setNumBounces(num_of_bounces);
	
	GpuRaytracer gpuRenderer;
	gpuRenderer.init(cudaCamera, cudaSpheres.data(), (int)cudaSpheres.size(), resolution, resolution);
	gpuRenderer.setNumBounces(num_of_bounces);
	
	GpuSharedRaytracer gpuSharedRenderer;
	gpuSharedRenderer.init(cudaCamera, cudaSpheres.data(), (int)cudaSpheres.size(), resolution, resolution);
	gpuSharedRenderer.setNumBounces(num_of_bounces);

	printf("\n=== CUDA Path Tracer Initialized ===\n");
	printf("Resolution: %d x %d\n", resolution, resolution);
	printf("CPU spheres: %zu, GPU spheres: %zu\n", cpuSpheres.size(), cudaSpheres.size());
	printf("Default mode: GPU (Shared Memory)\n\n");

	// Main rendering loop
	RenderModeUI lastRenderMode = render_mode;
	bool lastAccumulate = is_accumulation;
	int lastBounces = num_of_bounces;
	int lastThreads = num_of_threads;

	while (!glfwWindowShouldClose(window))
	{
		auto frameStart = std::chrono::high_resolution_clock::now();

		glfwPollEvents();
		
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		// Handle mode changes
		if (lastRenderMode != render_mode)
		{
			cpuRenderer.resetAccumulation();
			gpuRenderer.resetAccumulation();
			gpuSharedRenderer.resetAccumulation();
			lastRenderMode = render_mode;
		}

		// Handle accumulation changes
		if (lastAccumulate != is_accumulation)
		{
			cpuRenderer.setAccumulation(is_accumulation);
			gpuRenderer.setAccumulation(is_accumulation);
			gpuSharedRenderer.setAccumulation(is_accumulation);
			lastAccumulate = is_accumulation;
		}

		// Handle bounce changes
		if (lastBounces != num_of_bounces)
		{
			cpuRenderer.setNumBounces(num_of_bounces);
			gpuRenderer.setNumBounces(num_of_bounces);
			gpuSharedRenderer.setNumBounces(num_of_bounces);
			cpuRenderer.resetAccumulation();
			gpuRenderer.resetAccumulation();
			gpuSharedRenderer.resetAccumulation();
			lastBounces = num_of_bounces;
		}

		// Handle thread changes (CPU only)
		if (lastThreads != num_of_threads)
		{
			cpuRenderer.setNumThreads(num_of_threads);
			lastThreads = num_of_threads;
		}

		// Render using selected mode
		switch (render_mode)
		{
			case RenderModeUI::CPU:
				cpuRenderer.render(output_image_ptr);
				num_of_samples_rendered = cpuRenderer.getSampleCount();
				gpu_kernel_time_ms = 0.0f;
				gpu_memory_time_ms = 0.0f;
				break;
				
			case RenderModeUI::GPU_GLOBAL:
				gpuRenderer.render(output_image_ptr);
				{
					GpuRenderStats stats = gpuRenderer.getStats();
					num_of_samples_rendered = stats.samplesRendered;
					gpu_kernel_time_ms = stats.kernelTimeMs;
					gpu_memory_time_ms = stats.memoryTransferTimeMs;
				}
				break;
				
			case RenderModeUI::GPU_SHARED:
				gpuSharedRenderer.render(output_image_ptr);
				{
					GpuRenderStats stats = gpuSharedRenderer.getStats();
					num_of_samples_rendered = stats.samplesRendered;
					gpu_kernel_time_ms = stats.kernelTimeMs;
					gpu_memory_time_ms = stats.memoryTransferTimeMs;
				}
				break;
		}

		// Render UI
		glClear(GL_COLOR_BUFFER_BIT);
		ImGuiRender();
		glfwSwapBuffers(window);

		auto frameEnd = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> duration = frameEnd - frameStart;
		frame_seconds = duration.count();
	}

	// Cleanup
	cpuRenderer.cleanup();
	gpuRenderer.cleanup();
	gpuSharedRenderer.cleanup();
	
	delete[] output_image_ptr;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
