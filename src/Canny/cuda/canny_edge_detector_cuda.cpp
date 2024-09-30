//
// Created by Palnit on 2024. 01. 16.
//

#include "include/Canny/cuda/canny_edge_detector_cuda.h"
#include "include/Canny/cuda/cuda_canny_edge_detection.cuh"
#include "SDL_timer.h"
#include "include/general/OpenGL_SDL/basic_window.h"
#include "include/general/OpenGL_SDL/file_handling.h"

#include <cuda_runtime.h>

void CannyEdgeDetectorCuda::DetectEdge() {
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

    CudaCannyDetector detector(d_pixel,
                               m_base->w,
                               m_base->h,
                               m_gaussKernelSize,
                               m_standardDeviation,
                               m_highTrashHold,
                               m_lowTrashHold);
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
void CannyEdgeDetectorCuda::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}
void CannyEdgeDetectorCuda::DisplayImGui() {
    if (ImGui::BeginTabItem(m_name.c_str())) {

        if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 21)) {
            if (m_gaussKernelSize % 2 == 0) {
                m_gaussKernelSize++;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        ImGui::SliderFloat("Standard Deviation",
                           &m_standardDeviation,
                           0.0001f,
                           30.0f);
        ImGui::SliderFloat("High Trash Hold",
                           &m_highTrashHold,
                           0.0f,
                           255.0f);
        ImGui::SliderFloat("Low Trash Hold",
                           &m_lowTrashHold,
                           0.0f,
                           255.0f);
        ImGui::Separator();
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
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "CannyTimings:");
        ImGui::Text("Whole execution:         %f ms", m_timings.All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:            %f ms", m_timings.GrayScale_ms);
        ImGui::Text("Gauss Creation:          %f ms",
                    m_timings.GaussCreation_ms);
        ImGui::Text("Blur:                    %f ms", m_timings.Blur_ms);
        ImGui::Text("Sobel Operator:          %f ms",
                    m_timings.SobelOperator_ms);
        ImGui::Text("Non Maximum Suppression: %f ms",
                    m_timings.NonMaximumSuppression_ms);
        ImGui::Text("Double Threshold:        %f ms",
                    m_timings.DoubleThreshold_ms);
        ImGui::Text("Hysteresis:              %f ms", m_timings.Hysteresis_ms);

        ImGui::EndTabItem();
    }
}