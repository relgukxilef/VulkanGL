#include "enumerates.h"
#include "vulkan/vulkan_core.h"

#include <cstdio>

#include <GLES3/gl3.h>

const char *gl_error_string(GLenum error) {
    switch (error) {
        case GL_NO_ERROR: 
        case GL_FRAMEBUFFER_COMPLETE:
            return "No error has been recorded.";
        case GL_INVALID_ENUM: 
            return 
                "An unacceptable value is specified for an enumerated "
                "argument.";
        case GL_INVALID_VALUE:
            return "A numeric argument is out of range.";
        case GL_INVALID_OPERATION:
            return 
                "The specified operation is not allowed in the current state.";
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            return "The framebuffer object is not complete.";
        case GL_OUT_OF_MEMORY: 
            return "There is not enough memory left to execute the command.";
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 
                "At least one attachment point referenced by framebuffer is "
                "not framebuffer complete.";
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 
                "The framebuffer does not have at least one image attached to "
                "it for reading or writing.";
        case GL_FRAMEBUFFER_UNSUPPORTED:
            return 
                "The combination of internal formats of the attached images "
                "violates an implementation-dependent set of restrictions.";
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return "The attached images are not all the same size.";
        case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
            return 
                 "The value of FRAMEBUFFER_ATTACHMENT_TEXTURE_MULTISAMPLE is "
                 "not the same for all attached images.";
        default:
            return "An unknown error has been recorded.";
    }
}

void check(GLenum result) {
    if (result == GL_NO_ERROR || result == GL_FRAMEBUFFER_COMPLETE) {
        return;
    }
    fprintf(stderr, "OpenGL error. %s\n", gl_error_string(result));
}

format_info gl_format(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8_UNORM: return {GL_RGB8, 3, GL_UNSIGNED_BYTE};
    case VK_FORMAT_R8G8B8A8_UNORM: return {GL_RGBA8, 4, GL_UNSIGNED_BYTE};
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: 
        return {GL_RGB10_A2, 4, GL_UNSIGNED_INT_2_10_10_10_REV};
    case VK_FORMAT_D24_UNORM_S8_UINT: 
        return {GL_DEPTH24_STENCIL8, 1, GL_UNSIGNED_INT};
    case VK_FORMAT_R32G32B32_SFLOAT: return {GL_RGB32F, 3, GL_FLOAT};
    case VK_FORMAT_R32G32_SFLOAT: return {GL_RG32F, 2, GL_FLOAT};
    case VK_FORMAT_R8G8B8A8_SRGB: return {GL_RGBA8, 4, GL_UNSIGNED_BYTE};
    default:
        fprintf(stderr, "Unknown format %i\n", format);
        return {GL_R8, 1, GL_FLOAT};
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

GLenum gl_index_type(VkIndexType type) {
    switch (type) {
    case VK_INDEX_TYPE_UINT32:
        return GL_UNSIGNED_INT;
    case VK_INDEX_TYPE_UINT16:
        return GL_UNSIGNED_SHORT;
    case VK_INDEX_TYPE_UINT8_EXT:
        return GL_UNSIGNED_BYTE;
    default:
        fprintf(stderr, "Unknown index type %i\n", type);
        return GL_UNSIGNED_SHORT;
    }
};
