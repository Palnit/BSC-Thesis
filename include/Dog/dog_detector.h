#ifndef BSC_THESIS_DOG_DETECTOR_H
#define BSC_THESIS_DOG_DETECTOR_H

#include "Dog/OpenCl/dog_edge_detector_open_cl.h"
#include "general/OpenCL/get_devices.h"
#include "general/detector_base.h"
#include "imgui.h"

template<class T>
class DogDetector : public DetectorBase {
    static_assert(std::is_base_of<DogEdgeDetector, T>::value,
                  "Template type must have a base type of DogEdgeDetector");

public:
    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     * \param internal The internal name of the detector
     */
    DogDetector(SDL_Surface* base, std::string name, std::string internal)
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
        if (std::is_same_v<DogEdgeDetectorOpenCl, T>) {
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
        if (ImGui::SliderFloat("Standard Deviation 1",
                               m_detector.getStandardDeviation1(), 0.0001f,
                               30.0f)) {
            if (*m_detector.getStandardDeviation1()
                >= *m_detector.getStandardDeviation2()) {
                *m_detector.getStandardDeviation1() =
                    *m_detector.getStandardDeviation2() - 0.1f;
            }
        }
        ImGui::SetItemTooltip(
            "Standard Deviation 1 should be smaller than 2");
        if (ImGui::SliderFloat("Standard Deviation 2",
                               m_detector.getStandardDeviation2(), 0.0001f,
                               30.0f)) {
            if (*m_detector.getStandardDeviation1()
                >= *m_detector.getStandardDeviation2()) {
                *m_detector.getStandardDeviation2() =
                    *m_detector.getStandardDeviation1() + 0.1f;
            }
        }
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
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Dog Timings:");
        ImGui::Text("Whole execution:               %f ms",
                    m_detector.GetTimings().All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:                  %f ms",
                    m_detector.GetTimings().GrayScale_ms);
        ImGui::Text("Gauss 1 Creation:              %f ms",
                    m_detector.GetTimings().Gauss1Creation_ms);
        ImGui::Text("Gauss 2 Creation:              %f ms",
                    m_detector.GetTimings().Gauss1Creation_ms);
        ImGui::Text("Difference of gaussian:        %f ms",
                    m_detector.GetTimings().DifferenceOfGaussian_ms);
        ImGui::Text("Convolution:                   %f ms",
                    m_detector.GetTimings().Convolution_ms);
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

#endif//BSC_THESIS_DOG_DETECTOR_H
