#pragma once

#include <vulkan/vulkan.h>

VkInstance vglCreateInstanceForGL();
VkSurfaceKHR vglCreateSurfaceForGL();
void vglSetCurrentSurfaceExtent(VkExtent2D extent);