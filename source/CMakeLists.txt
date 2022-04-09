cmake_minimum_required(VERSION 3.5)

add_library(
    vulkangl STATIC
    vulkangl.cpp
)

target_link_libraries(
    vulkangl INTERFACE
    Vulkan-Headers
)