#include <cstdint>
#include <sys/types.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <memory>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <cstdio>
#include <cstring>
#include <vector>

#include <GLES3/gl3.h>
#include <spirv_glsl.hpp>

#include "globals.h"
#include "enumerates.h"

// TODO: move structs to header file
// TODO: move globals to namespace

struct VkPhysicalDevice_T {} global_physical_device;

struct VkDevice_T {
    GLuint copy_framebuffer;
} * global_device;
struct VkCommandPool_T {
    std::unique_ptr<struct VkCommandBuffer_T> buffers;
};
struct VkQueue_T {} global_queue;
struct VkSemaphore_T {};
struct VkImage_T {
    GLuint name;
    GLenum internal_format;
    GLsizei width, height, depth;
    GLsizei levels;
};
struct VkSwapchainKHR_T {
    VkImage_T image;
};
struct VkDebugUtilsMessengerEXT_T {};

struct VkFence_T {
    GLsync sync;
};

struct VkImageView_T {
    VkImage_T* image;
    VkImageSubresourceRange subresource_range;
};

struct VkShaderModule_T {
    // shader
    std::string glsl;
};

struct VkFramebuffer_T {
    GLuint name;
};

struct VkRenderPass_T {
};

struct command {
    virtual ~command() {};
    virtual void operator()() const = 0;
    std::unique_ptr<command> next;
};

template<class T>
struct lambda_command final : public command {
    lambda_command(T&& t) : t(t) {}
    void operator()() const override {
        t();
        (*next)();
    }
    T t;
};

struct stop_command : public command {
    void operator()() const override {}
};


struct VkPipeline_T {
    unsigned program;
};

struct VkCommandBuffer_T {
    const VkPipeline_T* graphics_pipeline;
    std::unique_ptr<VkCommandBuffer_T> buffer_next;

    struct {
        GLuint program;
        GLuint draw_framebuffer;
        GLuint read_framebuffer;
    } gl_state;

    std::unique_ptr<command> first, * next = &first;
};

template<class T>
void add_commmand(
    VkCommandBuffer commandBuffer, T&& functor
) {
    commandBuffer->next->reset(new lambda_command<T>(std::move(functor)));
    commandBuffer->next = &(*commandBuffer->next)->next;
}

VKAPI_ATTR void VKAPI_CALL
vkDestroySurfaceKHR(
    VkInstance instance, VkSurfaceKHR surface,
    const VkAllocationCallbacks* pAllocator
) {
}

VKAPI_ATTR VkResult VKAPI_CALL
vkEnumeratePhysicalDevices(
    VkInstance instance, uint32_t* pPhysicalDeviceCount,
    VkPhysicalDevice* pPhysicalDevices
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
    global_device = *pDevice;
    glGenFramebuffers(1, &(*pDevice)->copy_framebuffer);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator
) {
    global_device = nullptr;
    delete device;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(
    VkDevice device,
    const VkImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkImage* pImage
) {
    auto internal = new VkImage_T;
    glGenTextures(1, &internal->name);
    if (pCreateInfo->extent.depth == 1 && pCreateInfo->arrayLayers == 1) {
        glBindTexture(GL_TEXTURE_2D, internal->name);
        glTexStorage2D(
            GL_TEXTURE_2D, pCreateInfo->mipLevels, 
            gl_internal_format(pCreateInfo->format), 
            pCreateInfo->extent.width, pCreateInfo->extent.height
        );
    } else if (pCreateInfo->arrayLayers == 1) {
        glBindTexture(GL_TEXTURE_3D, internal->name);
        glTexStorage3D(
            GL_TEXTURE_3D, pCreateInfo->mipLevels, 
            gl_internal_format(pCreateInfo->format), 
            pCreateInfo->extent.width, pCreateInfo->extent.height, 
            pCreateInfo->extent.depth
        );
    } else {
        assert(false);
    }
    *pImage = (VkImage)internal;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImage(
    VkDevice device,
    VkImage image,
    const VkAllocationCallbacks* pAllocator
) {
    auto internal = (VkImage_T*)image;
    glDeleteTextures(1, &internal->name);
    delete internal;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(
    VkDevice device,
    const VkImageViewCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkImageView* pView
) {
    // TODO
    VkImageView_T* image_view = new VkImageView_T{
        .image = (VkImage_T*)pCreateInfo->image,
    };
    *pView = (VkImageView)image_view;
    
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(
    VkDevice device,
    VkImageView imageView,
    const VkAllocationCallbacks* pAllocator
) {
    VkImageView_T* internal_image_view = (VkImageView_T*)imageView;
    delete internal_image_view;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkShaderModule* pShaderModule
) {
    // At this point the shader stage is not known
    VkShaderModule_T shader_module;
    
    spirv_cross::CompilerGLSL compiler(
        pCreateInfo->pCode, pCreateInfo->codeSize / 4
    );

    spirv_cross::CompilerGLSL::Options options{
        .version = 300, // nothing newer supported in WebGL
        .es = true,
        .vertex = {
            .fixup_clipspace = true,
            .flip_vert_y = true,
        }
    };
	compiler.set_common_options(options);
    
    shader_module.glsl = compiler.compile();

    *pShaderModule = (VkShaderModule)new VkShaderModule_T(shader_module);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(
    VkDevice device,
    VkShaderModule shaderModule,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkShaderModule_T*)shaderModule;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
    VkDevice device,
    VkPipelineCache pipelineCache,
    uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo* pCreateInfos,
    const VkAllocationCallbacks* pAllocator,
    VkPipeline* pPipelines
) {
    for (int i = 0; i < createInfoCount; i++) {
        auto create_info = pCreateInfos[i];

        VkPipeline_T pipeline;
        pipeline.program = glCreateProgram();

        for (int j = 0; j < create_info.stageCount; j++) {
            auto stage = create_info.pStages[j];
            GLuint shader = glCreateShader(
                gl_shader_type(stage.stage)
            );
            auto shader_module = (VkShaderModule_T*)stage.module;
            const GLchar* source = shader_module->glsl.data();
            const GLint length = shader_module->glsl.size();
            glShaderSource(shader, 1, &source, &length);
            glCompileShader(shader);
            glAttachShader(pipeline.program, shader);
        }
        glLinkProgram(pipeline.program);
        pPipelines[i] = (VkPipeline)new VkPipeline_T(pipeline);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(
    VkDevice device,
    VkPipeline pipeline,
    const VkAllocationCallbacks* pAllocator
) {
    glDeleteProgram(((VkPipeline_T*)pipeline)->program);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkPipelineLayout* pPipelineLayout
) {
    // TODO
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(
    VkDevice device,
    VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks* pAllocator
) {
    // TODO
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(
    VkDevice device,
    const VkFramebufferCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkFramebuffer* pFramebuffer
) {
    auto framebuffer = new VkFramebuffer_T;
    // TODO: check whether this framebuffer is equivalent to framebuffer 0
    glGenFramebuffers(1, &framebuffer->name);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer->name);

    for (auto i = 0; i < pCreateInfo->attachmentCount; i++) {
        glBindTexture(
            GL_TEXTURE_2D, 
            ((VkImageView_T*)pCreateInfo->pAttachments[i])->image->name
        );
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,  GL_COLOR_ATTACHMENT0 + i, 	GL_TEXTURE_2D,
            ((VkImageView_T*)pCreateInfo->pAttachments[i])->image->name, 0
        );
    }

    auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
        fprintf(stderr, "Framebuffer has incomplete attachment");
    } else if (status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT) {
        fprintf(stderr, "Framebuffer needs at least one attachment");
    } else if (status == GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE) {
        fprintf(
            stderr,
            "Framebuffer attachments must have same number and "
            "location of samples"
        );
    } else if (status == GL_FRAMEBUFFER_UNSUPPORTED) {
        fprintf(stderr, "Framebuffer internal format not supported");
    } else if (status == GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS) {
        fprintf(
            stderr,
            "All attached images much have the same width and height."
        );
    } else if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer is incomplete for unknown reason");
    }

    *pFramebuffer = (VkFramebuffer)framebuffer;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(
    VkDevice device,
    VkFramebuffer framebuffer,
    const VkAllocationCallbacks* pAllocator
) {
    VkFramebuffer_T* internal_framebuffer = (VkFramebuffer_T*)framebuffer;
    if (internal_framebuffer->name != 0)
        glDeleteFramebuffers(1, &internal_framebuffer->name);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(
    VkDevice device,
    const VkRenderPassCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkRenderPass* pRenderPass
) {
    *pRenderPass = (VkRenderPass)new VkRenderPass_T;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(
    VkDevice device,
    VkRenderPass renderPass,
    const VkAllocationCallbacks* pAllocator
) {
    delete (VkFramebuffer_T*)renderPass;
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
            .format = VK_FORMAT_R8G8B8_UNORM,
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
        auto buffer = std::make_unique<VkCommandBuffer_T>();
        buffer->buffer_next = 
            std::move(((VkCommandPool_T*)pAllocateInfo->commandPool)->buffers);
        pCommandBuffers[i] = buffer.get();
        ((VkCommandPool_T*)pAllocateInfo->commandPool)->buffers = 
            std::move(buffer);
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
    *commandBuffer->next = std::make_unique<stop_command>();
    commandBuffer->next = &(**commandBuffer->next).next;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipeline pipeline
) {
    GLuint program = ((VkPipeline_T*)pipeline)->program;
    if (commandBuffer->gl_state.program != program) {
        add_commmand(commandBuffer, [=](){
            glUseProgram(program);
        });
        commandBuffer->gl_state.program = program;
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(
    VkCommandBuffer commandBuffer,
    uint32_t firstViewport,
    uint32_t viewportCount,
    const VkViewport* pViewports
) {
    add_commmand(commandBuffer, [=](){
        glViewport(
            pViewports[0].x, pViewports[0].y, 
            pViewports[0].width, pViewports[0].height
        );
        glDepthRangef(pViewports[0].minDepth, pViewports[0].maxDepth);
    });
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(
    VkCommandBuffer commandBuffer,
    uint32_t firstScissor,
    uint32_t scissorCount,
    const VkRect2D* pScissors
) {
    add_commmand(commandBuffer, [=](){
        glScissor(
            pScissors[0].offset.x, pScissors[0].offset.y, 
            pScissors[0].extent.width, pScissors[0].extent.height
        );
    });
}

VKAPI_ATTR void VKAPI_CALL vkCmdDraw(
    VkCommandBuffer commandBuffer,
    uint32_t vertexCount,
    uint32_t instanceCount,
    uint32_t firstVertex,
    uint32_t firstInstance
) {
    if (instanceCount == 1) {
        add_commmand(commandBuffer, [=](){
            glDrawArrays(GL_TRIANGLES, firstVertex, vertexCount);
        });
    } else if (firstInstance == 0) {
        add_commmand(commandBuffer, [=](){
            glDrawArraysInstanced(
                GL_TRIANGLES, firstVertex, vertexCount, instanceCount
            );
        });
    } else {
        // TODO
    }
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

VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(
    VkCommandBuffer commandBuffer,
    const VkRenderPassBeginInfo* pRenderPassBegin,
    VkSubpassContents contents
) {
    GLuint framebuffer = 
        ((VkFramebuffer_T*)pRenderPassBegin->framebuffer)->name;
    auto color = pRenderPassBegin->pClearValues->color;

    if (commandBuffer->gl_state.draw_framebuffer != framebuffer) {
        add_commmand(commandBuffer, [framebuffer, color](){ 
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glClearColor(
                color.float32[0], color.float32[1], 
                color.float32[2], color.float32[3]
            );
            glClear(GL_COLOR_BUFFER_BIT);
        });
        commandBuffer->gl_state.draw_framebuffer = framebuffer;

    } else {
        add_commmand(commandBuffer, [color](){ 
            glClearColor(
                color.float32[0], color.float32[1], 
                color.float32[2], color.float32[3]
            );
            glClear(GL_COLOR_BUFFER_BIT);
        });
    }
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(
    VkCommandBuffer commandBuffer
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
    GLuint name;
    glGenTextures(1, &name);
    glBindTexture(GL_TEXTURE_2D, name);
    glTexStorage2D(
        GL_TEXTURE_2D, 1, 
        GL_RGB8, 
        pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height
    );
    *pSwapchain = (VkSwapchainKHR)new VkSwapchainKHR_T{
        .image = {
            .name = name,
            .internal_format = gl_internal_format(pCreateInfo->imageFormat),
            .width = current_surface_extent.width,
            .height = current_surface_extent.height,
            .depth = 1,
        }
    };
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* pAllocator
) {
    auto internal = (VkSwapchainKHR_T*)swapchain;
    glDeleteTextures(1, &internal->image.name);
    delete internal;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    uint32_t* pSwapchainImageCount,
    VkImage* pSwapchainImages
) {
    if (pSwapchainImages && *pSwapchainImageCount >= 1) {
        pSwapchainImages[0] = (VkImage)&((VkSwapchainKHR_T*)swapchain)->image;
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
        .currentExtent = { 
            .width = (uint32_t)current_surface_extent.width, 
            .height = (uint32_t)current_surface_extent.height 
        },
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

VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    uint64_t timeout,
    VkSemaphore semaphore,
    VkFence fence,
    uint32_t* pImageIndex
) {
    auto internal_swapchain = (VkSwapchainKHR_T*)swapchain;
    if (
        internal_swapchain->image.width == current_surface_extent.width &&
        internal_swapchain->image.height == current_surface_extent.height
    ) {
        *pImageIndex = 0;
        return VK_SUCCESS;

    } else {
        return VK_ERROR_OUT_OF_DATE_KHR;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(
    VkQueue queue,
    uint32_t submitCount,
    const VkSubmitInfo* pSubmits,
    VkFence fence
) {
    for (
        const VkSubmitInfo* submit = pSubmits; submit != pSubmits + submitCount;
        submit++
    ) {
        for (
            const VkCommandBuffer* buffer = submit->pCommandBuffers; 
            buffer != submit->pCommandBuffers + submit->commandBufferCount;
            buffer++
        ) {
            // TODO: handle semaphores and fences
            (*(*buffer)->first)();
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* pPresentInfo
) {
    // TODO: handle semaphores
    // TODO: handle multiple swapchains?
    if (pPresentInfo->swapchainCount > 0) {
        auto image = ((VkSwapchainKHR_T*)pPresentInfo->pSwapchains[0])->image;
        if (
            current_surface_extent.width == image.width && 
            current_surface_extent.height == image.height
        ) {
            // TODO: optimize away copy
            glBindFramebuffer(
                GL_READ_FRAMEBUFFER, global_device->copy_framebuffer
            );
            glFramebufferTexture2D(
                GL_READ_FRAMEBUFFER,  GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                image.name, 0
            );
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(
                0, 0, image.width, image.height, 
                0, 0, image.width, image.height,
                GL_COLOR_BUFFER_BIT, GL_NEAREST
            );
            return VK_SUCCESS;
        
        } else {
            // TODO: can resize during rendering even happen?
            return VK_ERROR_OUT_OF_DATE_KHR;
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(
    VkDevice device,
    uint32_t fenceCount,
    const VkFence* pFences
) {
    // TODO
    return VK_SUCCESS;
}
