#include <cassert>

#include "vulkan/vulkan_core.h"

#include <GLES3/gl3.h>

GLenum gl_internal_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8_UNORM: return GL_RGB8;
    case VK_FORMAT_R8G8B8A8_UNORM: return GL_RGBA8;
    default:
        assert(false);
        return GL_R8;
    }
};