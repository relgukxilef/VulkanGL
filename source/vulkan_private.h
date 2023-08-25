#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <GLES3/gl3.h>

struct VkPhysicalDevice_T {
    VkPhysicalDeviceProperties device_properties;
};

struct gl_extent_2d {
    GLsizei width, height;
};

struct VkDevice_T {
    GLuint copy_framebuffer;
};

struct VkQueue_T {};

namespace vgl {
    extern VkPhysicalDevice_T* global_physical_device;
    extern gl_extent_2d current_surface_extent;
    extern VkDevice_T* global_device;
    extern VkQueue_T global_queue;
    extern VkDeviceSize device_memory, host_memory;
}
