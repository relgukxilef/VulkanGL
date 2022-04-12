#include "vulkangl/vulkangl.h"

struct VkInstance_T {} global_instance;

VkInstance vglCreateInstanceForGL() {
    return &global_instance;
}

VkSurfaceKHR vglCreateSurfaceForGL() {
    return 1;
}