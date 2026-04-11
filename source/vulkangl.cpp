#include "vulkangl/vulkangl.h"

#include "vulkan_private.h"

VkInstance vglCreateInstanceForGL() {
    return (VkInstance)&vgl::global_instance;
}

VkSurfaceKHR vglCreateSurfaceForGL() {
    return (VkSurfaceKHR)&vgl::global_surface;
}

void vglSetCurrentSurfaceExtent(VkExtent2D extent) {
    vgl::current_surface_extent = {
        (GLsizei)extent.width, 
        (GLsizei)extent.height 
    };
}

void vglSetDeviceMemory(VkDeviceSize size) {
    vgl::device_memory = size;
}

void vglSetHostMemory(VkDeviceSize size) {
    vgl::host_memory = size;
}
