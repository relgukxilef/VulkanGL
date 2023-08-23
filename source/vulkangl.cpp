#include "vulkangl/vulkangl.h"

#include "vulkan_private.h"
#include <utility>
#include <cstring>

struct VkInstance_T {} global_instance;
struct VkSurfaceKHR_T {} global_surface;

VkInstance vglCreateInstanceForGL() {
    global_physical_device = new VkPhysicalDevice_T;
    global_physical_device->device_properties = VkPhysicalDeviceProperties{
        .apiVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
        .driverVersion = 1,
        .vendorID = 0,
        .deviceID = 0,
        .deviceType = VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
        .deviceName = {},
        .pipelineCacheUUID = {},
        .limits = {}, // TODO
        .sparseProperties = {},
    };

    auto vendor = (const char*)glGetString(GL_VENDOR);
    auto renderer = (const char*)glGetString(GL_RENDERER);
    auto vendor_length = std::strlen(vendor);
    auto renderer_length = std::strlen(renderer);

    auto& name = global_physical_device->device_properties.deviceName;
    memcpy(
        name, vendor,
        std::min<size_t>(vendor_length, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE)
    );
    name[vendor_length] = ' ';
    memcpy(
        name + vendor_length + 1, renderer,
        std::min<size_t>(
            renderer_length,
            VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - vendor_length - 1
        )
    );

    return &global_instance;
}

VkSurfaceKHR vglCreateSurfaceForGL() {
    return (VkSurfaceKHR)&global_surface;
}

void vglSetCurrentSurfaceExtent(VkExtent2D extent) {
    current_surface_extent = { 
        (GLsizei)extent.width, 
        (GLsizei)extent.height 
    };
}
