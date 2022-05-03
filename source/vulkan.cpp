#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "globals.h"

struct VkPhysicalDevice_T {} global_physical_device;

struct VkDevice_T {};
struct VkCommandPool_T {};
struct VkQueue_T {} global_queue;
struct VkSemaphore_T {};
struct VkSwapchainKHR_T {};
struct VkDebugUtilsMessengerEXT_T {};
struct VkImage_T {} default_image;

struct VkFence_T {
    GLsync sync;
};

struct command {
    virtual ~command();
    virtual void operator()();
    command* next;
};

template<class T>
struct lambda_command : public command {
    lambda_command(T&& t) : t(t) {}
    void operator()() {
        t();
        (*next)();
    }
    T t;
};

struct stop_command : public command {
    void operator()() {}
};


struct VkPipeline_T {
    unsigned program;
};

struct VkCommandBuffer_T {
    VkPipeline_T* graphics_pipeline;

    struct {
        unsigned program;
    } gl_state;

    command* first, ** next = &first;
};

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
            // no compute on WebGL yet
            .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT,
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
    *pDevice = new VkDevice_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator
) {
    delete device;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
    VkDevice device,
    const VkCommandPoolCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkCommandPool* pCommandPool
) {
    *pCommandPool = (VkCommandPool)new VkCommandPool_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(
    VkDevice device,
    VkCommandPool commandPool,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkCommandPool_T*)commandPool;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(
    VkDevice device,
    uint32_t queueFamilyIndex,
    uint32_t queueIndex,
    VkQueue* pQueue
) {
    *pQueue = &global_queue;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR surface,
    uint32_t* pSurfaceFormatCount,
    VkSurfaceFormatKHR* pSurfaceFormats
) {
    if (pSurfaceFormats && *pSurfaceFormatCount >= 1) {
        pSurfaceFormats[0] = {
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .colorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR
        };
    }
    *pSurfaceFormatCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR surface,
    uint32_t* pPresentModeCount,
    VkPresentModeKHR* pPresentModes
) {
    if (pPresentModes && *pPresentModeCount >= 1) {
        pPresentModes[0] = VK_PRESENT_MODE_FIFO_KHR;
    }
    *pPresentModeCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
    VkDevice device,
    const VkCommandBufferAllocateInfo* pAllocateInfo,
    VkCommandBuffer* pCommandBuffers
) {
    for (unsigned i = 0; i < pAllocateInfo->commandBufferCount; ++i) {
        pCommandBuffers[i] = new VkCommandBuffer_T;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(
    VkCommandBuffer commandBuffer,
    const VkCommandBufferBeginInfo* pBeginInfo
) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(
    VkCommandBuffer commandBuffer
) {
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags srcStageMask,
    VkPipelineStageFlags dstStageMask,
    VkDependencyFlags dependencyFlags,
    uint32_t memoryBarrierCount,
    const VkMemoryBarrier* pMemoryBarriers,
    uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier* pBufferMemoryBarriers,
    uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* pImageMemoryBarriers
) {

}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(
    VkDevice device,
    const VkFenceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkFence* pFence
) {
    *pFence = (VkFence)new VkFence_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFence(
    VkDevice device,
    VkFence fence,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkFence_T*)fence;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(
    VkDevice device,
    uint32_t fenceCount,
    const VkFence* pFences,
    VkBool32 waitAll,
    uint64_t timeout
) {
    glFinish();
    /*
    auto end = 
        std::chrono::high_resolution_clock::now() + 
        std::chrono::nanoseconds(timeout);
    for (unsigned i = 0; i < fenceCount; ++i) {
        auto fence = (VkFence_T*)pFences[i];
        // TODO: maybe add OpenGL-Registry to the submodules?
        auto result = 
            glClientWaitSyncAPPLE(
                fence->sync, GL_SYNC_FLUSH_COMMANDS_BIT_APPLE, timeout
            );

        if (result == GL_TIMEOUT_EXPIRED_APPLE) {
            return VK_TIMEOUT;
        }
        
        timeout = std::max(std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - std::chrono::high_resolution_clock::now()
        ).count(), 0);
    }*/
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(
    VkDevice device,
    const VkSemaphoreCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkSemaphore* pSemaphore
) {
    *pSemaphore = (VkSemaphore)new VkSemaphore_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(
    VkDevice device,
    VkSemaphore semaphore,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkSemaphore_T*)semaphore;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
    VkDevice device,
    const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkSwapchainKHR* pSwapchain
) {
    *pSwapchain = (VkSwapchainKHR)new VkSwapchainKHR_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkSwapchainKHR_T*)swapchain;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    uint32_t* pSwapchainImageCount,
    VkImage* pSwapchainImages
) {
    if (pSwapchainImages && *pSwapchainImageCount >= 1) {
        pSwapchainImages[0] = (VkImage)&default_image;
    }
    *pSwapchainImageCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR surface,
    VkSurfaceCapabilitiesKHR* pSurfaceCapabilities
) {
    *pSurfaceCapabilities = {
        .minImageCount = 1,
        .maxImageCount = 1,
        .currentExtent = current_surface_extent,
        .minImageExtent = {
            .width = 1,
            .height = 1
        },
        .maxImageExtent = {
            .width = 0xffffffff,
            .height = 0xffffffff
        },
        .maxImageArrayLayers = 1,
        .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    };
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT*   pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger
) {
    *pMessenger = (VkDebugUtilsMessengerEXT)new VkDebugUtilsMessengerEXT_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkDebugUtilsMessengerEXT_T*)messenger;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance instance,
    const char* pName
) {
    if (strcmp(pName, "vkCreateDebugUtilsMessengerEXT") == 0) {
        return (PFN_vkVoidFunction)vkCreateDebugUtilsMessengerEXT;
    }
    
    if (strcmp(pName, "vkDestroyDebugUtilsMessengerEXT") == 0) {
        return (PFN_vkVoidFunction)vkDestroyDebugUtilsMessengerEXT;
    }
    return nullptr;
}
