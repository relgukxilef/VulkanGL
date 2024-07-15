cmake_minimum_required(VERSION 3.5)

add_library(
    vulkangl STATIC
    vulkangl.cpp
    vulkan.cpp
    enumerates.h enumerates.cpp
    egl.c gles2.c
    vulkan_private.h
)

target_link_libraries(
    vulkangl PUBLIC
    Vulkan-Headers
    spirv-cross-glsl
)
