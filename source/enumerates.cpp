#include "enumerates.h"
#include "vulkan/vulkan_core.h"

#include <cstdio>

#include <GLES3/gl3.h>

GLenum gl_internal_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8_UNORM:
        return GL_RGB8;
    case VK_FORMAT_R8G8B8A8_UNORM:
        return GL_RGBA8;
    default:
        fprintf(stderr, "Unknown format %i\n", format);
        return GL_R8;
    }
};

GLenum gl_shader_type(VkShaderStageFlagBits stage) {
    switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
        return GL_VERTEX_SHADER;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
        return GL_FRAGMENT_SHADER;
    default:
        fprintf(stderr, "Unknown shader stage %i\n", stage);
        return GL_VERTEX_SHADER;
    }
};

GLenum gl_primitive_type(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
        return GL_TRIANGLES;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        return GL_TRIANGLE_STRIP;
    default:
        fprintf(stderr, "Unknown primitive topology %i\n", topology);
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
};
