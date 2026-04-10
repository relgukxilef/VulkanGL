cmake_minimum_required(VERSION 3.5)

project(vulkangl LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.307
)
FetchContent_Declare(
    SPIRV-Cross
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
    GIT_TAG vulkan-sdk-1.4.341.0
)
FetchContent_MakeAvailable(
    VulkanHeaders SPIRV-Cross
)

add_subdirectory(source)

add_library(VulkanGL::VulkanGL ALIAS vulkangl)

target_include_directories(
    vulkangl PUBLIC
    include
)

target_compile_features(vulkangl PRIVATE cxx_std_20)

if (EMSCRIPTEN)
#    target_link_options(
#        vulkangl PRIVATE -sMIN_WEBGL_VERSION=2 -sMAX_WEBGL_VERSION=2
#    )
    target_compile_definitions(vulkangl PRIVATE EMSCRIPTEN)
endif()
