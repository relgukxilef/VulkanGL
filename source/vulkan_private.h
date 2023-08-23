#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <GLES3/gl3.h>

struct VkPhysicalDevice_T {
    VkPhysicalDeviceProperties device_properties;
};
extern VkPhysicalDevice_T* global_physical_device;

struct gl_extent_2d {
    GLsizei width, height;
};

extern gl_extent_2d current_surface_extent;
