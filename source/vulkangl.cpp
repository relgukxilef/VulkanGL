#include "vulkangl/vulkangl.h"

struct VkInstance_T {} global_instance;
struct VkSurfaceKHR_T {} global_surface;

VkInstance vglCreateInstanceForGL() {
    return &global_instance;
}

VkSurfaceKHR vglCreateSurfaceForGL() {
    return (VkSurfaceKHR)&global_surface;
}