#Specify the version being used as well as the language
cmake_minimum_required(VERSION 3.30)

#Name your project here
project(helloworld LANGUAGES C CXX CUDA)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include Dirs: ${CUDAToolkit_INCLUDE_DIRS}")
# OpenGL is a very popular 3D rendering API for raster graphics
find_package(OpenGL REQUIRED)

# GLFW is a popular window manager that works well with OpenGL
find_package(glfw3 CONFIG REQUIRED)

# GLM is a light-weight linear algebra library optimized for 2D and 3D graphics
find_package(glm CONFIG REQUIRED)

# ImGui is a real-time user interface that works with both OpenGL and GLFW
#	This requires three ImGui packages: core, glfw-binding, and opengl3-binding
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)

# GLEW is a library used to find the function signatures for OpenGL extensions
find_package(GLEW REQUIRED)

# I usually set a few parameters when building in Visual Studio since the defaults
#	can make things difficult. What this mostly does is set the Debug and Release
#	directories to the build directory. That way the executable is always in the same
#	location when testing the application from the command line.
#
# Another thing this does is enable the ability to use functions like printf() which
#	are deprecated and may cause warnings.
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
elseif( APPLE )
# These settings are used for Apple systems (MacOS). It sets agressive optimization
#	for Release builds and makes sure that the Cocoa, IOKit, and CoreVideo frameworks
#	are linked (which is required for GLFW).
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
else()
# These settings are used for GCC and other compilers (usually on Linux systems). It sets
#	agressive optimization for Release builds and makes sure that the X11 libraries are
#	linked (which is required for GLFW).
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ( MSVC )

# Set all of the include directories so that you can include the necessary header files
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${GLFW_INCLUDE_DIRS}
			${TIRA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
			JPEG::JPEG
)

file(GLOB_RECURSE SOURCE
	src/*.cpp src/*.h src/*.cu src/*.cuh
)

set(CMAKE_CUDA_ARCHITECTURES 61)
# Set the files required to build the executable
add_executable(helloworld
	${SOURCE}
)


set_target_properties(helloworld PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 20
    CXX_STANDARD 20
)

# Set all of the libraries so that the linker knows where to find them
target_link_libraries(helloworld
PRIVATE 
	glm::glm
	glfw
	GLEW::GLEW
	imgui::imgui
	CUDA::cudart
)