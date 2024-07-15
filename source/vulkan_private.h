#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <GLES3/gl3.h>

struct VkPhysicalDevice_T {
    VkPhysicalDeviceProperties device_properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
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

struct memory_type_info {
    GLenum default_binding;
    VkBufferUsageFlags buffer_usage;
    VkImageUsageFlags image_usage;
};

const memory_type_info memory_types[] = {
    {
        GL_ARRAY_BUFFER, 
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
        0
    }, { // ELEMENT_ARRAY_BUFFER in WebGL can not be bound to other targets
        GL_ELEMENT_ARRAY_BUFFER,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        0
    }, {
        GL_NONE,
        0,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT
    },
};
