//
// Created by Palnit on 2024. 01. 22.
//

#include <chrono>
#include "include/Dog/cpu/dog_edge_detector_cpu.h"
#include "imgui.h"
#include "include/general/cpu/gauss_blur_cpu.h"
#include "SDL_image.h"

void DogEdgeDetectorCPU::DetectEdge() {
    m_pixels1 =
        static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    m_pixels2 =
        static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    m_kernel1 = static_cast<float*>(malloc(
        sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    m_kernel2 = static_cast<float*>(malloc(
        sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    m_finalKernel = static_cast<float*>(malloc(
        sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));

    auto t1 = std::chrono::high_resolution_clock::now();

    m_timings.GrayScale_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::ConvertGrayScale,
                                  (uint8_t*) m_base->pixels,
                                  m_pixels1,
                                  m_w,
                                  m_h);
    m_timings.Gauss1Creation_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel1,
                                  m_gaussKernelSize,
                                  m_standardDeviation1);
    m_timings.Gauss2Creation_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel2,
                                  m_gaussKernelSize,
                                  m_standardDeviation2);
    m_timings.DifferenceOfGaussian_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::DifferenceOfGaussian,
                                  m_kernel1,
                                  m_kernel2,
                                  m_finalKernel,
                                  m_gaussKernelSize);

    m_timings.Convolution_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::GaussianFilter, m_pixels1,
                                  m_pixels2,
                                  m_finalKernel,
                                  m_gaussKernelSize,
                                  m_w,
                                  m_h);

    DetectorsCPU::CopyBack((uint8_t*) m_detected->pixels,
                           m_pixels2,
                           m_w,
                           m_h);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    m_timingsReady = true;
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
    free(m_pixels1);
    free(m_pixels2);
    free(m_kernel1);
    free(m_kernel2);
    free(m_finalKernel);

}

void DogEdgeDetectorCPU::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}

void DogEdgeDetectorCPU::DisplayImGui() {

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
void DetectorsCPU::DifferenceOfGaussian(float* kernel1,
                                        float* kernel2,
                                        float* finalKernel,
                                        int kernelSize) {
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            *(finalKernel + i + (j * kernelSize)) =
                *(kernel1 + i + (j * kernelSize))
                    - *(kernel2 + i + (j * kernelSize));
        }
    }
}
