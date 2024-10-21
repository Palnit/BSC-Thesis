
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TEST_INCLUDES_FILES
        include/spiral_indexer.h
        include/surface_painters.h
        include/testing_window.h
        include/testing_imgui_display.h
        include/tester_base.h
        include/Canny/Cpu/canny_cpu_tester.h
        )

set(TEST_SOURCE_FILES
        src/main.cpp
        src/surface_painters.cpp
        src/spiral_indexer.cpp
        src/testing_window.cpp
        src/testing_imgui_display.cpp
        src/tester_base.cpp
        src/Canny/Cpu/canny_cpu_tester.cpp
        )


add_executable(BSC_Thesis_Tests
        ${TEST_SOURCE_FILES}
        ${TEST_HEADER_FILES}
        )


target_compile_features(BSC_Thesis_Tests PRIVATE cxx_std_17)

target_link_libraries(BSC_Thesis_Tests PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>
        imgui::imgui
        GLEW::GLEW
        glm::glm
        OpenGL::GL
        implot::implot
        OpenCL::OpenCL
        ${CMAKE_PROJECT_NAME}_lib
        )


target_include_directories(BSC_Thesis_Tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )


find_package(CUDA)
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND AND CUDA_FOUND)
    enable_language(CUDA)

    target_link_libraries(BSC_Thesis_Tests PRIVATE
            cuda_comp
            CUDA::toolkit)
endif ()
