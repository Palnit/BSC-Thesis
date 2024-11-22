cmake_minimum_required(VERSION 3.22)

include(FetchContent)
FetchContent_Declare(vcpkg
        GIT_REPOSITORY "https://github.com/microsoft/vcpkg/"
)
FetchContent_MakeAvailable(vcpkg)

set(CMAKE_TOOLCHAIN_FILE "${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
set(VCPKG_TRACE_FIND_PACKAGE true)

project(BSC_Thesis
        VERSION 0.1.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HEADER_FILES
        include/general/OpenGL_SDL/basic_window.h
        include/general/OpenGL_SDL/SDL_GLDebugMessageCallback.h
        include/general/main_window.h
        include/general/detector_base.h
        include/general/OpenGL_SDL/file_handling.h
        include/general/OpenGL_SDL/generic_structs.h
        include/general/OpenGL_SDL/vertex_buffer_object.h
        include/general/OpenGL_SDL/vertex_array_object.h
        include/general/OpenGL_SDL/element_buffer_object.h
        include/general/OpenGL_SDL/shader_program.h
        include/general/imgui_display.h
        include/Canny/canny_timings.h
        include/Dog/dog_timings.h
        include/general/cpu/gauss_blur_cpu.h
        include/general/cpu/morphology_cpu.h
        include/general/edge_detector_base.h
        include/general/timings_base.h
        include/Canny/canny_edge_detector.h
        include/Canny/cpu/canny_edge_detector_cpu.h
        include/Canny/canny_detector.h
        include/Dog/dog_detector.h
        include/Dog/cpu/dog_edge_detector_cpu.h)

set(SOURCE_FILES
        src/general/OpenGL_SDL/basic_window.cpp
        src/general/OpenGL_SDL/SDL_GLDebugMessageCallback.cpp
        src/general/main_window.cpp
        src/general/detector_base.cpp
        src/general/OpenGL_SDL/file_handling.cpp
        src/general/OpenGL_SDL/generic_structs.cpp
        src/general/OpenGL_SDL/vertex_buffer_object.cpp
        src/general/OpenGL_SDL/vertex_array_object.cpp
        src/general/OpenGL_SDL/element_buffer_object.cpp
        src/general/OpenGL_SDL/shader_program.cpp
        src/general/imgui_display.cpp
        src/general/cpu/gauss_blur_cpu.cpp
        src/general/cpu/morphology_cpu.cpp
        src/Canny/cpu/canny_edge_detector_cpu.cpp
        src/Dog/cpu/dog_edge_detector_cpu.cpp)

set(CUDA_HEADER_FILES
        include/Canny/cuda/cuda_canny_edge_detection.cuh
        include/general/cuda/gauss_blur.cuh
        include/Dog/cuda/cuda_dog_edge_detection.cuh
)

set(CUDA_CPP_HEADER_FILES
        include/Canny/cuda/canny_detector_cuda.h
        include/Dog/cuda/dog_detector_cuda.h
        include/general/cuda/gpu_info.h
)

set(CUDA_SOURCE_FILES
        src/Canny/cuda/cuda_canny_edge_detection.cu
        src/general/cuda/gauss_blur.cu
        src/Dog/cuda/cuda_dog_edge_detection.cu
)

set(CUDA_CPP_SOURCE_FILES
        src/Canny/cuda/canny_detector_cuda.cpp
        src/general/cuda/gpu_info.cpp
        src/Dog/cuda/dog_detector_cuda.cpp
)

set(OPENCL_HEADER_FILES
        include/general/OpenCL/get_devices.h
        include/general/OpenCL/file_handling.h
        include/general/OpenCL/program.h
        include/general/OpenCL/memory.h
        include/general/OpenCL/kernel.h
        include/Canny/OpenCl/canny_edge_detector_open_cl.h
        include/Dog/OpenCl/dog_edge_detector_open_cl.h
)

set(OPENCL_SOURCE_FILES
        src/general/OpenCL/get_devices.cpp
        src/general/OpenCL/file_handling.cpp
        src/general/OpenCL/program.cpp
        src/general/OpenCL/kernel.cpp
        src/general/OpenCL/memory.cpp
        src/Canny/OpenCl/canny_edge_detector_open_cl.cpp
        src/Dog/OpenCl/dog_edge_detector_open_cl.cpp)

set(RESOURCES
        pictures/img.jpg
        pictures/test.jpg
        pictures/d2.png
        pictures/d2_90.jpg
        pictures/github.PNG
        pictures/d4.gif
        pictures/base2.png

        shaders/default_fragment.frag
        shaders/default_vertex.vert

        OpenCLKernels/gauss_blur.cl
        OpenCLKernels/canny.cl
        OpenCLKernels/dog.cl

)

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/win/version.rc.in
            ${CMAKE_CURRENT_BINARY_DIR}/win/version.rc
            @ONLY)

    enable_language("RC")
    set(WIN32_RESOURCES
            ${CMAKE_CURRENT_BINARY_DIR}/win/version.rc
            ${CMAKE_CURRENT_SOURCE_DIR}/win/BSC_Thesis.exe.manifest
    )
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX})
    set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX})
    set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX})
else ()
    include(GNUInstallDirs)
endif ()

find_package(CUDA)
find_package(CUDAToolkit)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_image CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(implot CONFIG REQUIRED)
find_package(OpenCL CONFIG REQUIRED)


if (CUDAToolkit_FOUND AND CUDA_FOUND)
    enable_language(CUDA)
    add_compile_definitions(CUDA_EXISTS)
    add_library(cuda_comp STATIC
            ${CUDA_HEADER_FILES}
            ${CUDA_SOURCE_FILES}
    )
    target_compile_features(cuda_comp PRIVATE
            cuda_std_17
    )

    set_target_properties(cuda_comp PROPERTIES LINKER_LANGUAGE CUDA)

    target_link_libraries(cuda_comp PRIVATE
            CUDA::toolkit
            CUDA::cudart
            CUDA::cuda_driver
    )

    set_target_properties(cuda_comp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    target_include_directories(cuda_comp
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(cuda_comp
            PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_STANDARD 17
            CUDA_SEPARABLE_COMPILATION OFF
    )
endif ()

add_executable(${CMAKE_PROJECT_NAME}
        src/main.cpp
        ${WIN32_RESOURCES}
)

add_library(${CMAKE_PROJECT_NAME}_lib STATIC
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${OPENCL_SOURCE_FILES}
        ${OPENCL_HEADER_FILES}
)
if (CUDAToolkit_FOUND AND CUDA_FOUND)
    target_sources(${CMAKE_PROJECT_NAME}_lib PRIVATE
            ${CUDA_CPP_HEADER_FILES}
            ${CUDA_CPP_SOURCE_FILES}
    )
endif ()

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_features(${CMAKE_PROJECT_NAME}_lib PRIVATE cxx_std_17)


target_link_libraries(${CMAKE_PROJECT_NAME}_lib PRIVATE
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>
        imgui::imgui
        GLEW::GLEW
        glm::glm
        OpenGL::GL
        implot::implot
        OpenCL::OpenCL
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_PROJECT_NAME}_lib
        $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
        $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
        $<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>
        imgui::imgui
        GLEW::GLEW
        glm::glm
        OpenGL::GL
        implot::implot
        OpenCL::OpenCL
)

if (CUDAToolkit_FOUND AND CUDA_FOUND)

    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
            cuda_comp
            CUDA::toolkit)
    target_link_libraries(${CMAKE_PROJECT_NAME}_lib PRIVATE
            cuda_comp
            CUDA::toolkit)
endif ()

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(${CMAKE_PROJECT_NAME}_lib PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${CMAKE_PROJECT_NAME}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)


target_include_directories(${CMAKE_PROJECT_NAME}_lib
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(test)


foreach (file ${RESOURCES})
    string(REPLACE "/" "_" n_file ${file})
    ADD_CUSTOM_TARGET(
            copy_${n_file}_file
            ${CMAKE_COMMAND}
            -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -D DESTINATION_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -D file=${file}
            -P copy.cmake
            COMMENT "Running custom target for file: ${file} in project: ${CMAKE_PROJECT_NAME}" VERBATIM
    )
    ADD_DEPENDENCIES(${CMAKE_PROJECT_NAME} copy_${n_file}_file)
endforeach ()


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/copy.cmake DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
