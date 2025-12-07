#include "helloworld.h"

// This variable stores the OpenGL ID for the texture used to display the output image on the screen
GLuint output_image_tex = 0;

/*
 * This function uploads the current image stored on the heap to the GPU so that it can be displayed on the screen. This
 * function is called every frame, so any updates to the heap image will be displayed when the frame updates. All of these
 * function calls are part of the OpenGL API.
 */
void UpdateOutputTexture()
{
	/* OpenGL textures are given integer IDs starting at 1, so here we test to see if the ID is zero (in which case a
     * texture hasn't been created. If that's the case, this code generates a new texture ID.
     */
	if (output_image_tex == 0)
		glGenTextures(1, &output_image_tex);

	/* The OpenGL API is a "state machine", which means that you make global function calls to query and set parts of that
     * state. Since there can be multiple textures on the GPU, the first thing we have to do is tell OpenGL which texture
     * we want future OpenGL API calls to reference. The term for that is "binding" the texture. So here I'm just
     * setting the texture I just created as the "current" texture, so future calls will update that texture's state.
     */
	glBindTexture(GL_TEXTURE_2D, output_image_tex);

	/* These calls update the bound texture's state. All these are doing is telling OpenGL how to treat that texture.
     * Specifically, the first two functions say what happens if we try to access regions "outside" of the texture
     * These two functions just describe what value is returned if we access elements of the texture
     * that are "between" pixels. For example, if I access pixel (3.5, 2.5). GL_NEAREST just says to set the returned
     * value to the nearest pixel. This can make the image look "pixellated" if you zoom in really close, but I prefer
     * it for scientific applications because it gives a better idea of what the data really looks like. In video games
     * you'll usually use linear interpolation (GL_LINEAR).
     */
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	/* This function actually uploads the texture to the GPU. The parameters specify:
     * 1) What texture target we are uploading (in this case the currently "bound" 2D texture)
     * 2) The "mip-map" level of the image. I'm not using that feature so this is just 0
     * 3) How the texture data will be stored on the GPU (in this case as a 4-channel RGBA image)
     * 4/5) The size of the image in pixels
     * 6) The size of the "border" around the image (there isn't one so this is 0)
     * 7) The format of the pixel data that I'm sending to the GPU (also a 4-channel RGBA image)
     * 8) The data type used to store the pixel data (we're using 8-but unsigned integers)
     * 9) A pointer to the pixel data to upload to the GPU
     */
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resolution, resolution, 0, GL_RGBA, GL_FLOAT, output_image_ptr);
}

/* This function is called every time the user interface is updated (which happens every iteration of the main loop).
 * This displays the contents of the current output image on the screen in an ImGui window.
 */
void DrawOutputImage()
{
	// Update the OpenGL texture (on the GPU) with the current image data in main memory
	UpdateOutputTexture();

	// Create a new ImGui window to show the image - call it whatever you want
	ImGui::Begin("Hello World!");

	// Get the size of the window region that can be drawn to
	ImVec2 viewport_size = ImGui::GetContentRegionAvail();

	// Use ImGui to render the image to the screen (we've set up ImGui to use OpenGL, so we provide an OpenGL texture to render)
	ImGui::Image((ImTextureID)(intptr_t)output_image_tex, viewport_size);

	// This is the only thing displayed in the window
	ImGui::End();
}