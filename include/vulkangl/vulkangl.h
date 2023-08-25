#pragma once

#include <vulkan/vulkan.h>

VkInstance vglCreateInstanceForGL();
VkSurfaceKHR vglCreateSurfaceForGL();
void vglSetCurrentSurfaceExtent(VkExtent2D extent);
void vglSetDeviceMemory(VkDeviceSize size);
void vglSetHostMemory(VkDeviceSize size);
