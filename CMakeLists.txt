cmake_minimum_required(VERSION 3.5)

project(vulkangl LANGUAGES CXX)

add_subdirectory(source)
add_subdirectory(test)

add_subdirectory(submodules/Vulkan-Headers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
