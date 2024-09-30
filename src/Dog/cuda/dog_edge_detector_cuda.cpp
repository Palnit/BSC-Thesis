//
// Created by Palnit on 2024. 01. 21.
//

#include <cuda_runtime.h>
#include "include/Dog/cuda/dog_edge_detector_cuda.h"
#include "imgui.h"
#include "include/Dog/cuda/cuda_dog_edge_detection.cuh"
#include "SDL_image.h"

void DogEdgeDetectorCuda::DetectEdge() {
    uint8_t* d_pixel = nullptr;

    cudaMalloc((void**) &d_pixel,
               sizeof(uint8_t) * m_base->w
                   * m_base->h
                   * m_base->format->BytesPerPixel);

    cudaMemcpy(d_pixel,
               m_base->pixels,
               sizeof(uint8_t) * m_base->w * m_base->h
                   * m_base->format->BytesPerPixel,
               cudaMemcpyHostToDevice);

    CudaDogDetector detector(d_pixel,
                             m_base->w,
                             m_base->h,
                             m_gaussKernelSize,
                             m_standardDeviation1,
                             m_standardDeviation2);
    m_timings = detector.GetTimings();
    m_timingsReady = true;

    cudaMemcpy(m_detected->pixels,
               d_pixel,
               sizeof(uint8_t) * m_detected->w * m_detected->h
                   * m_detected->format->BytesPerPixel,
               cudaMemcpyDeviceToHost);

    cudaFree(d_pixel);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_detected->w,
                 m_detected->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_detected->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}
void DogEdgeDetectorCuda::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}
void DogEdgeDetectorCuda::DisplayImGui() {

    if (ImGui::BeginTabItem(m_name.c_str())) {

        if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 21)) {
            if (m_gaussKernelSize % 2 == 0) {
                m_gaussKernelSize++;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        if (ImGui::SliderFloat("Standard Deviation 1",
                               &m_standardDeviation1,
                               0.0001f,
                               30.0f)) {
            if (m_standardDeviation1 >= m_standardDeviation2) {
                m_standardDeviation1--;
            }
        }
        ImGui::SetItemTooltip("Standard Deviation 1 should be smaller than 2");
        if (ImGui::SliderFloat("Standard Deviation 2",
                               &m_standardDeviation2,
                               0.0001f,
                               30.0f)) {
            if (m_standardDeviation1 >= m_standardDeviation2) {
                m_standardDeviation2++;
            }
        }
        if (ImGui::Button("Detect")) {
            DetectEdge();
        }
        if (!m_timingsReady) {
            ImGui::EndTabItem();
            return;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            std::string save_path = "./" + m_name + ".png";
            IMG_SavePNG(m_detected, save_path.c_str());
        }

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "DogTimings:");
        ImGui::Text("Whole execution:               %f ms", m_timings.All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:                  %f ms",
                    m_timings.GrayScale_ms);
        ImGui::Text("Gauss 1 Creation:              %f ms",
                    m_timings.Gauss1Creation_ms);
        ImGui::Text("Gauss 2 Creation:              %f ms",
                    m_timings.Gauss1Creation_ms);
        ImGui::Text("Difference of gaussian:        %f ms",
                    m_timings.DifferenceOfGaussian_ms);
        ImGui::Text("Convolution:                   %f ms",
                    m_timings.Convolution_ms);
        ImGui::EndTabItem();
    }
}
