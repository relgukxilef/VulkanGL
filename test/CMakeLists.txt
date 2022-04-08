cmake_minimum_required(VERSION 3.5)

add_executable(
    test_gl
    main.cpp
)

target_link_libraries(
    test_gl PRIVATE
    glfw3dll glew32s opengl32 vulkangl
)

add_executable(
    test_vulkan
    main.cpp
)

target_link_libraries(
    test_vulkan PRIVATE
    vulkan-1
)
