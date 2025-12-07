#include "helloworld.h"

#include <thread>

/*
 * This function is a starting point for creating your own user interface. I just create a UI window and
 * add a timer. Other elements can be added by putting them between ImGui::Begin() and ImGui::End()
 */
void DrawInterfaceWindow()
{
	// Create a new ImGui window to show the image - call it whatever you want
	ImGui::Begin("Render Settings");

	// Display the render time for a single pass through the main loop
	ImGui::Text("Resolution: %d x %d", resolution, resolution);
	ImGui::Text("Frame Time: %.2f ms (%.1f FPS)", frame_seconds * 1000, 1.0f / frame_seconds);
	ImGui::Text("Samples Rendered: %d", num_of_samples_rendered);

	ImGui::Separator();
	ImGui::Text("=== Render Mode ===");
	
	// Render mode selection
	const char* modeNames[] = { "CPU (Multi-threaded)", "GPU (Global Memory)", "GPU (Shared Memory)" };
	int currentMode = static_cast<int>(render_mode);
	
	if (ImGui::Combo("Mode", &currentMode, modeNames, 3))
	{
		render_mode = static_cast<RenderModeUI>(currentMode);
	}
	
	// Show mode-specific info
	if (render_mode == RenderModeUI::CPU)
	{
		ImGui::Text("Using CPU with %d threads", num_of_threads);
		ImGui::Text("Max available threads: %d", std::thread::hardware_concurrency());
	}
	else
	{
		ImGui::Text("GPU Kernel Time: %.2f ms", gpu_kernel_time_ms);
		ImGui::Text("GPU Memory Transfer: %.2f ms", gpu_memory_time_ms);
		if (render_mode == RenderModeUI::GPU_SHARED)
		{
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Shared Memory: ENABLED");
		}
		else
		{
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Shared Memory: DISABLED");
		}
	}

	ImGui::Separator();
	ImGui::Text("=== Accumulation ===");
	
	// Accumulation toggle
	ImGui::Checkbox("Accumulate Samples", &is_accumulation);

	ImGui::Separator();
	ImGui::Text("=== Ray Tracing Settings ===");

	// Number of bounces
	ImGui::SliderInt("Bounces", &num_of_bounces, 1, 16);
	
	// CPU-specific settings
	if (render_mode == RenderModeUI::CPU)
	{
		ImGui::Separator();
		ImGui::Text("=== CPU Settings ===");
		
		// Number of threads
		int maxThreads = std::thread::hardware_concurrency();
		ImGui::SliderInt("Threads", &num_of_threads, 1, maxThreads);
	}

	ImGui::End();
}

/*
* This function renders the user interface. I'm actually cheating a little bit here: the only user
* interface window that's rendered is a "demo" that comes with the ImGui library. It basically has a
* bunch of widgets that show what ImGui is capable of, so you have some interesting stuff to play with
* and I didn't actually have to program any of it.
* 
* In any case, you can add your own user interface elements here.
*/
void ImGuiRender()
{
	// These functions initialize the UI rendering process with both OpenGL and GLFW
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();

	// This function creates a new "Frame", which is the basic foundation of an ImGui UI
	ImGui::NewFrame();
	{
		/* This renders an ImGui "Demo" window that shows off its UI elements (you can delete this and replace it with your own)
		 * You can get an equivalent of this window online, which also provides the code necessary to create each UI element:
		 * https://pthom.github.io/imgui_manual_online/manual/imgui_manual.html
		 */
		ImGui::ShowDemoWindow();

		// This renders an ImGui window displaying the output image
		DrawOutputImage();

		// Draw a placeholder user interface window (you can use this function or add additional windows with similar functions)
		DrawInterfaceWindow();
	}

	// This function makes the graphics API calls (in this case OpenGL) to render the user interface
	ImGui::Render();

	// This actually copies the GUI to the OpenGL frame buffer (in this case probably the GLFW back buffer)
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}