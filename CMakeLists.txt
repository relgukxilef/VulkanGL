cmake_minimum_required(VERSION 3.5)

project(vulkangl LANGUAGES CXX)

add_subdirectory(source)
add_subdirectory(submodules/Vulkan-Headers)

target_include_directories(
    vulkangl PUBLIC
    include
)

target_compile_features(vulkangl PRIVATE cxx_std_11)