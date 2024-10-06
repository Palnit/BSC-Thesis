set(TEST_INCLUDES_FILES)

set(TEST_SOURCE_FILES
    src/main.cpp
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


if (CUDAToolkit_FOUND AND CUDA_FOUND)
    enable_language(CUDA)

    target_link_libraries(BSC_Thesis_Tests PRIVATE
            cuda_comp
            CUDA::toolkit)
endif ()
