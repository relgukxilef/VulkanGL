#include <cassert>

#include "vulkan/vulkan_core.h"

#include <GLES3/gl3.h>

struct format_info {
    GLenum internal_format;
    GLint size;
    GLenum type;
};

void check(GLenum result);

format_info gl_format(VkFormat format);

GLenum gl_shader_type(VkShaderStageFlagBits);

GLenum gl_primitive_type(VkPrimitiveTopology topology);

GLenum gl_index_type(VkIndexType type);
