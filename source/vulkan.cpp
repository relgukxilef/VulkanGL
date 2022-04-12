#include <vulkan/vulkan.h>

#include <iostream>

#include <GLES2/gl2.h>

VKAPI_ATTR void VKAPI_CALL
vkDestroySurfaceKHR(
    VkInstance instance, VkSurfaceKHR surface,
    const VkAllocationCallbacks *pAllocator
) {
    std::cout << "vkDestroySurfaceKHR" << std::endl;
}