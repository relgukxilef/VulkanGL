#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

VkInstance vglCreateInstanceForGL();
VkSurfaceKHR vglCreateSurfaceForGL();
void vglSetCurrentSurfaceExtent(VkExtent2D extent);
void vglSetDeviceMemory(VkDeviceSize size);
void vglSetHostMemory(VkDeviceSize size);
VkImage vglVkImageFromGL(
    uint32_t image, uint32_t internal_format, int32_t width, int32_t height,
    int32_t depth
);
VkFormat vglVkFormatFromGL(uint32_t internal_format);
