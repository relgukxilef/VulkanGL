#include "vulkangl/vulkangl.h"

#include "vulkan_private.h"
#include "enumerates.h"

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

VkImage vglVkImageFromGL(
    uint32_t image, uint32_t internal_format, int32_t width, int32_t height, 
    int32_t depth
) {
    return new VkImage_T{
        .target = GL_TEXTURE_2D,
        .texture = image,
        .internal_format = internal_format,
        .width = width,
        .height = height,
        .depth = depth,
    };
}

VkFormat vglVkFormatFromGL(GLuint internal_format) {
    return vk_format(internal_format);
}
