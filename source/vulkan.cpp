#include <vulkan/vulkan.h>

#include <GLES2/gl2.h>

struct VkPhysicalDevice_T {} global_physical_device;
struct VkDevice_T {} global_device;

VKAPI_ATTR void VKAPI_CALL
vkDestroySurfaceKHR(
    VkInstance instance, VkSurfaceKHR surface,
    const VkAllocationCallbacks *pAllocator
) {
}

VKAPI_ATTR VkResult VKAPI_CALL
vkEnumeratePhysicalDevices(
    VkInstance instance, uint32_t *pPhysicalDeviceCount,
    VkPhysicalDevice *pPhysicalDevices
) {
    if (pPhysicalDevices && *pPhysicalDeviceCount >= 1) {
        pPhysicalDevices[0] = &global_physical_device;
    } 
    *pPhysicalDeviceCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* pQueueFamilyPropertyCount,
    VkQueueFamilyProperties* pQueueFamilyProperties
) {
    if (pQueueFamilyProperties && *pQueueFamilyPropertyCount >= 1) {
        pQueueFamilyProperties[0] = {
            .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
            .queueCount = 1,
            .timestampValidBits = 0,
            .minImageTransferGranularity = {1, 1, 1}
        };
    }
    *pQueueFamilyPropertyCount = 1;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice physicalDevice,
    uint32_t queueFamilyIndex,
    VkSurfaceKHR surface,
    VkBool32* pSupported
) {
    *pSupported = VK_TRUE;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice
) {
    *pDevice = &global_device;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator
) {
}
