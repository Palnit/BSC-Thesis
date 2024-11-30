#ifndef BSC_THESIS_CANNY_DETECTOR_H
#define BSC_THESIS_CANNY_DETECTOR_H

#include <fstream>
#include "SDL_image.h"
#include "canny_edge_detector.h"
#include "chrono"
#include "general/OpenCL/get_devices.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "general/cpu/gauss_blur_cpu.h"
#include "general/detector_base.h"
#include "imgui.h"

template<class T>
class CannyDetector : public DetectorBase {
    static_assert(std::is_base_of<CannyEdgeDetector, T>::value,
                  "Template type must have a base type of CannyEdgeDetector");

public:
    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     * \param internal The internal name of the detector
     */
    CannyDetector(SDL_Surface* base, std::string name, std::string internal)
        : DetectorBase(base, std::move(name), std::move(internal)) {}

    /*!
     * Implementation of the Display function displays the base and
     * detected image
     */
    void Display() override {
        shaderProgram.Bind();
        VAO.Bind();
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
        VAO.UnBind();
        shaderProgram.UnBind();
    }

    /*!
     * Implementation of the MainWindowDisplayImGui function displays the variables
     * related to this edge detection method to be modified easily
     */
    void DisplayImGui() override {
        if (std::is_same_v<CannyEdgeDetectorOpenCl, T>) {
            if (OpenCLInfo::OPENCL_DEVICES[0]
                .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                < 1024) {
                ImGui::Text("Not Enough Work Group");
                return;
            }
        }
        std::string text = "Detector Options for " + m_name;
        ImGui::SeparatorText(text.c_str());

        if (ImGui::SliderInt("Gauss Kernel Size",
                             m_detector.getGaussKernelSize(), 3, 21)) {
            if (*m_detector.getGaussKernelSize() % 2 == 0) {
                *m_detector.getGaussKernelSize() += 1;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        ImGui::SliderFloat("Standard Deviation",
                           m_detector.getStandardDeviation(), 0.0001f,
                           30.0f);
        ImGui::SliderFloat("High ThreshHold", m_detector.getHigh(), 0.0f,
                           255.0f);
        ImGui::SliderFloat("Low ThreshHold", m_detector.getLow(), 0.0f,
                           255.0f);
        ImGui::Separator();
        if (ImGui::Button("Detect")) { DetectEdge(); }
        if (!m_timingsReady) {
            return;
        }
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            std::string save_path = "./" + m_internalName + ".png";
            IMG_SavePNG(m_detected, save_path.c_str());
        }
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "CannyTimings:");
        ImGui::Text("Whole execution:         %f ms",
                    m_detector.GetTimings().All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:            %f ms",
                    m_detector.GetTimings().GrayScale_ms);
        ImGui::Text("Gauss Creation:          %f ms",
                    m_detector.GetTimings().GaussCreation_ms);
        ImGui::Text("Blur:                    %f ms",
                    m_detector.GetTimings().Blur_ms);
        ImGui::Text("Sobel Operator:          %f ms",
                    m_detector.GetTimings().SobelOperator_ms);
        ImGui::Text("Non Maximum Suppression: %f ms",
                    m_detector.GetTimings().NonMaximumSuppression_ms);
        ImGui::Text("Double Threshold:        %f ms",
                    m_detector.GetTimings().DoubleThreshold_ms);
        ImGui::Text("Hysteresis:              %f ms",
                    m_detector.GetTimings().Hysteresis_ms);

    }

    void DetectEdge() override {
        m_detector.setH(m_base->h);
        m_detector.setW(m_base->w);
        m_detector.setPixels((uint8_t*) m_base->pixels);
        m_detector.setStride(m_base->format->BytesPerPixel);
        auto detected = m_detector.Detect();
        m_timingsReady = true;

        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_base->w, m_base->h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, detected.get());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        std::copy(detected.get(),
                  detected.get()
                      + (m_base->w * m_base->h * m_base->format->BytesPerPixel),
                  (uint8_t*) m_detected->pixels);
    }

protected:
    T m_detector;
};

#endif//BSC_THESIS_CANNY_DETECTOR_H
