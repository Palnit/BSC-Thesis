
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TEST_INCLUDES_FILES
        include/spiral_indexer.h
        include/surface_painters.h
        include/testing_window.h
        include/testing_imgui_display.h
        include/tester_base.h
        include/Dog/dog_tester.h
        include/Canny/canny_tester.h
)

set(TEST_SOURCE_FILES
        src/main.cpp
        src/surface_painters.cpp
        src/spiral_indexer.cpp
        src/testing_window.cpp
        src/testing_imgui_display.cpp
)

set(TEST_RESOURCES
        OpenCLKernels/canny.cl
        OpenCLKernels/dog.cl
        OpenCLKernels/gauss_blur.cl
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/../win/version.rc.in
            ${CMAKE_CURRENT_BINARY_DIR}/win/version.rc
            @ONLY)

    enable_language("RC")
    set(WIN32_RESOURCES_TESTS
            ${CMAKE_CURRENT_BINARY_DIR}/win/version.rc
            ${CMAKE_CURRENT_SOURCE_DIR}/../win/BSC_Thesis.exe.manifest
    )
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX})
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX})
    set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX})
else ()
    include(GNUInstallDirs)
endif ()

add_executable(BSC_Thesis_Tests
        ${TEST_SOURCE_FILES}
        ${TEST_INCLUDES_FILES}
        ${WIN32_RESOURCES_TESTS}
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
    add_compile_definitions(CUDA_EXISTS)

    target_link_libraries(BSC_Thesis_Tests PRIVATE
            cuda_comp
            CUDA::toolkit)
endif ()


foreach (file ${TEST_RESOURCES})
    string(REPLACE "/" "_" n_file ${file})
    ADD_CUSTOM_TARGET(
            test_copy_${n_file}_file
            ${CMAKE_COMMAND}
            -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
            -D DESTINATION_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -D file=${file}
            -P copy.cmake
            COMMENT "Running custom target for file: ${file} in project: ${CMAKE_PROJECT_NAME}" VERBATIM
    )
    ADD_DEPENDENCIES(BSC_Thesis_Tests test_copy_${n_file}_file)
endforeach ()


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../copy.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
