#include "vulkangl/vulkangl.h"

#include "vulkan_private.h"
#include <algorithm>
#include <cstring>

struct VkInstance_T {} global_instance;
struct VkSurfaceKHR_T {} global_surface;

VkInstance vglCreateInstanceForGL() {
    VkPhysicalDeviceMemoryProperties properties{
        .memoryTypeCount = std::size(memory_types),
        .memoryHeapCount = 1,
    };
    for (auto i = 0u; i < properties.memoryTypeCount; i++)
        properties.memoryTypes[i] = VkMemoryType{
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            0
        };
    properties.memoryHeaps[0] = {
        .size = vgl::device_memory,
        .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
    };
    
    vgl::global_physical_device = new VkPhysicalDevice_T{
        .device_properties = {
            .apiVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
            .driverVersion = 1,
            .vendorID = 0,
            .deviceID = 0,
            .deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            .deviceName = {},
            .pipelineCacheUUID = {},
            .limits = {
                .nonCoherentAtomSize = 4,
            }, // TODO
            .sparseProperties = {},
        },
        .memory_properties = properties,
    };

    return &global_instance;
}

VkSurfaceKHR vglCreateSurfaceForGL() {
    return (VkSurfaceKHR)&global_surface;
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
