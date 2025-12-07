#pragma once

// OpenGL Extension Wrangler - declares functions used to find OpenGL function declarations
#include <GL/glew.h>

// GLFW - declares functions used for the GLFW window manager
#include <GLFW/glfw3.h>

// ImGui - declares functions used for the ImGui user interface
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// used to time events (currently only the main loop is timed)
#include <chrono>

// Render mode enumeration (matches cuda_raytracer.cuh)
enum class RenderModeUI {
    CPU,            // CPU multi-threaded rendering
    GPU_GLOBAL,     // GPU rendering with global memory only
    GPU_SHARED      // GPU rendering with shared memory optimization
};

extern float* output_image_ptr;
extern int	  resolution;
extern float  frame_seconds;
extern int	  num_of_samples_rendered;
extern bool	  is_accumulation;
extern int	  num_of_threads;
extern int	  num_of_bounces;
extern RenderModeUI render_mode;  // Current render mode (CPU, GPU_GLOBAL, GPU_SHARED)

// Performance stats
extern float  gpu_kernel_time_ms;
extern float  gpu_memory_time_ms;

void ImGuiRender();
void DrawOutputImage();
void UpdateOutputTexture();