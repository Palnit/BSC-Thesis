//
// Created by Palnit on 2024. 01. 21.
//

#include "include/Canny/cpu/canny_edge_detector_cpu.h"
#include "imgui.h"
#include "include/general/OpenGL_SDL/generic_structs.h"
#include "include/general/cpu/gauss_blur_cpu.h"
#include "chrono"
#include "SDL_image.h"

void CannyEdgeDetectorCPU::DetectEdge() {
    m_pixels1 =
        static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    m_pixels2 =
        static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    m_kernel = static_cast<float*>(malloc(
        sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    m_tangent =
        static_cast<float*>(malloc(sizeof(float) * m_w * m_h));

    auto t1 = std::chrono::high_resolution_clock::now();
    m_timings.GrayScale_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::ConvertGrayScale,
                                  (uint8_t*) m_base->pixels,
                                  m_pixels1,
                                  m_w,
                                  m_h);
    m_timings.GaussCreation_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel,
                                  m_gaussKernelSize,
                                  m_standardDeviation);

    m_timings.Blur_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::GaussianFilter, m_pixels1,
                                  m_pixels2,
                                  m_kernel,
                                  m_gaussKernelSize,
                                  m_w,
                                  m_h);
    m_timings.SobelOperator_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::DetectionOperator,
                                  m_pixels2,
                                  m_pixels1,
                                  m_tangent,
                                  m_w,
                                  m_h);
    m_timings.NonMaximumSuppression_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::NonMaximumSuppression,
                                  m_pixels1,
                                  m_pixels2,
                                  m_pixels2,
                                  m_w,
                                  m_h);
    m_timings.DoubleThreshold_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::DoubleThreshold,
                                  m_pixels2,
                                  m_pixels1,
                                  m_w,
                                  m_h,
                                  m_highTrashHold,
                                  m_lowTrashHold);
    m_timings.Hysteresis_ms =
        DetectorsCPU::TimerRunner(DetectorsCPU::Hysteresis,
                                  m_pixels1,
                                  m_pixels2,
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
    free(m_tangent);
}

void CannyEdgeDetectorCPU::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}

void CannyEdgeDetectorCPU::DisplayImGui() {
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
void DetectorsCPU::DetectionOperator(float* src,
                                     float* dest,
                                     float* tangent,
                                     int w,
                                     int h) {
    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};
    int k = 1;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            float SumX = 0;
            float SumY = 0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    int ix = x + i;
                    int jx = y + j;
                    if (ix < 0) {
                        ix = 0;
                    }
                    if (ix >= w) {
                        ix = w - 1;
                    }
                    if (jx < 0) {
                        jx = 0;
                    }
                    if (jx >= h) {
                        jx = h - 1;
                    }
                    SumX += *(src + ix + (jx * w))
                        * (*(SobelX + (i + k) + ((j + k) * 3)));
                    SumY += *(src + ix + (jx * w))
                        * (*(SobelY + (i + k) + ((j + k) * 3)));

                }
            }
            *(dest + x + (y * w)) = hypotf(SumX, SumY);
            float angle = (atan2(SumX, SumY) * 180.f) / M_PI;
            if (angle < 0) {
                angle += 180;
            }
            *(tangent + x + (y * w)) = angle;
        }
    }

}
void DetectorsCPU::NonMaximumSuppression(float* src,
                                         float* dest,
                                         float* tangent,
                                         int w,
                                         int h) {
    float* tangentA;
    float gradientA;
    float gradientP;
    float gradientN;
    int yp;
    int yn;
    int xp;
    int xn;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {

            tangentA = (tangent + x + (y * w));
            gradientA = *(src + x + (y * w));
            gradientP = 2000;
            gradientN = 2000;

            yp = y + 1;
            if (yp >= h) {
                yp = h - 1;
            }
            xp = x + 1;
            if (xp >= h) {
                xp = h - 1;
            }
            yn = y - 1;
            if (yn < 0) {
                yn = 0;
            }
            xn = x - 1;
            if (xn < 0) {
                xn = 0;
            }

            if ((0 <= *tangentA && *tangentA < 22.5)
                || (157.5 <= *tangentA && *tangentA <= 180)) {
                gradientP = *(src + x + (yp * w));
                gradientN = *(src + x + (yn * w));
            } else if (22.5 <= *tangentA && *tangentA < 67.5) {
                gradientP = *(src + xp + (yn * w));
                gradientN = *(src + xn + (yp * w));
            } else if (67.5 <= *tangentA && *tangentA < 112.5) {
                gradientP = *(src + xp + (y * w));
                gradientN = *(src + xn + (y * w));
            } else if (112.5 <= *tangentA && *tangentA < 157.5) {
                gradientP = *(src + xn + (yn * w));
                gradientN = *(src + xp + (yp * w));
            }

            if (gradientA < gradientN || gradientA < gradientP) {
                gradientA = 0.f;
            }
            *(dest + x + (y * w)) = gradientA;
        }
    }
}
void DetectorsCPU::DoubleThreshold(float* src,
                                   float* dest,
                                   int w,
                                   int h,
                                   float high,
                                   float low) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (*(src + x + (y * w)) >= high) {
                *(dest + x + (y * w)) = 255.f;
            } else if (*(src + x + (y * w)) < high
                && *(src + x + (y * w)) >= low) {
                *(dest + x + (y * w)) = 125.f;
            } else {
                *(dest + x + (y * w)) = 0.f;
            }
        }
    }
}
void DetectorsCPU::Hysteresis(float* src, float* dest, int w, int h) {
    int k = 1;
    bool strong = false;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            float SumX = 0;
            float SumY = 0;
            *(dest + x + (y * w)) = *(src + x + (y * w));
            if (*(src + x + (y * w)) != 125.f) {
                continue;
            }
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    int ix = x + i;
                    int jy = y + j;
                    if (ix < 0) {
                        ix = 0;
                    }
                    if (ix >= w) {
                        ix = w - 1;
                    }
                    if (jy < 0) {
                        jy = 0;
                    }
                    if (jy >= h) {
                        jy = h - 1;
                    }

                    if (*(src + ix + (jy * w)) == 255.f) {
                        strong = true;
                    }
                }
            }
            if (strong) {
                *(dest + x + (y * w)) = 255.f;
            } else {
                *(dest + x + (y * w)) = 0;
            }
        }
    }
}
