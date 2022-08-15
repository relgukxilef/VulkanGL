cmake_minimum_required(VERSION 3.5)

add_library(
    vulkangl STATIC
    vulkangl.cpp
    vulkan.cpp
    globals.h globals.cpp
    enumerates.h enumerates.cpp
)

target_link_libraries(
    vulkangl PUBLIC
    Vulkan-Headers
    spirv-cross-glsl
)
