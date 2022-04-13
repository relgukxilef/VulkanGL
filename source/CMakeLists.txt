cmake_minimum_required(VERSION 3.5)

add_library(
    vulkangl STATIC
    vulkangl.cpp
    vulkan.cpp
)

target_link_libraries(
    vulkangl PUBLIC
    Vulkan-Headers
)