#ifndef BSC_THESIS_TEST_INCLUDE_CANNY_CANNY_TESTER_H_
#define BSC_THESIS_TEST_INCLUDE_CANNY_CANNY_TESTER_H_

#include <fstream>
#include <random>
#include <numeric>
#include "tester_base.h"
#include "Canny/canny_edge_detector.h"
#include "Canny/OpenCl/canny_edge_detector_open_cl.h"
#include "general/OpenCL/get_devices.h"
#include "implot.h"
#include "SDL_surface.h"
#include "surface_painters.h"
#include "SDL_image.h"
#include "spiral_indexer.h"

template<class T>
class CannyTester : public TesterBase {
    static_assert(std::is_base_of<CannyEdgeDetector, T>::value,
                  "Template type must have a base type of DogEdgeDetector");

public:
    explicit CannyTester(const std::string& name,
                         const std::string& internalName)
        : TesterBase(name, internalName) {
        *m_detector.getHigh() = 20;
        *m_detector.getLow() = 10;
    }
    void ResultDisplay() override {
        if (m_selected) {
            if (std::is_same_v<CannyEdgeDetectorOpenCl, T>) {
                if (OpenCLInfo::OPENCL_DEVICES[0]
                    .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                    < 1024) {
                    ImGui::Text("Not Enough Work Group");
                    return;
                }
            }
            auto* x = new float[m_iterations];
            auto* x2 = new int[m_iterations];
            for (int i = 0; i < m_iterations; i++) { x[i] = x2[i] = i; }
            if (ImGui::BeginTabBar("Errors")) {
                if (ImGui::BeginTabItem("Error Rate")) {
                    if (ImPlot::BeginPlot("Error Rate")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Avg closest true pixel",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, m_AVG.data(), m_AVG.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Missing")) {
                    if (ImPlot::BeginPlot("Missing")) {
                        ImPlot::SetupAxes(
                            "Iteration", "No real pixel found",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x2, m_missing.data(),
                                         m_missing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            if (ImGui::BeginTabBar("Timings")) {
                if (ImGui::BeginTabItem("Whole execution")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.All_ms);
                    }
                    if (ImPlot::BeginPlot("Whole execution")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Gray Scaling")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.GrayScale_ms);
                    }
                    if (ImPlot::BeginPlot("Gray Scaling")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Gauss Creation")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.GaussCreation_ms);
                    }
                    if (ImPlot::BeginPlot("Gauss Creation")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Blur")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.Blur_ms);
                    }
                    if (ImPlot::BeginPlot("Blur")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Sobel Operator")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.SobelOperator_ms);
                    }
                    if (ImPlot::BeginPlot("Sobel Operator")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Non Maximum Suppression")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.NonMaximumSuppression_ms);
                    }
                    if (ImPlot::BeginPlot("Non Maximum Suppression")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Double Threshold")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.DoubleThreshold_ms);
                    }
                    if (ImPlot::BeginPlot("Double Threshold")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Hysteresis")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.Hysteresis_ms);
                    }
                    if (ImPlot::BeginPlot("Hysteresis")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
            if (!m_missing.empty() && !m_AVG.empty()) {
                ImGui::Text("Avg missing: %f",
                            std::reduce(m_missing.begin(), m_missing.end())
                                / (float) m_missing.size());
                ImGui::SameLine();
                ImGui::Text("Avg error: %f",
                            std::reduce(m_AVG.begin(), m_AVG.end())
                                / (float) m_AVG.size());
            }
            delete[] x;
            delete[] x2;

        }

    }
    void SpecializedDisplayImGui() override {
        if (std::is_same_v<CannyEdgeDetectorOpenCl, T>) {
            if (OpenCLInfo::OPENCL_DEVICES[0]
                .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                < 1024) {
                ImGui::Text("Not Enough Work Group");
                return;
            }
        }
        ImGui::SeparatorText("Canny Settings");
        if (ImGui::SliderInt("Gauss Kernel Size",
                             m_detector.getGaussKernelSize(), 3, 21)) {
            if (*m_detector.getGaussKernelSize() % 2 == 0) {
                *m_detector.getGaussKernelSize() += 2;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        ImGui::SliderFloat("Standard Deviation",
                           m_detector.getStandardDeviation(), 0.0001f,
                           30.0f);
        ImGui::SliderFloat("High Threshold", m_detector.getHigh(), 0.0f,
                           255.0f);
        ImGui::SliderFloat("Low Threshold", m_detector.getLow(), 0.0f,
                           255.0f);
    }
    void Test() override {
        m_AVG.clear();
        m_allTimings.clear();
        m_missing.clear();

        RGBA colorNormal = {static_cast<uint8_t>(m_backGroundColor[0]),
                            static_cast<uint8_t>(m_backGroundColor[1]),
                            static_cast<uint8_t>(m_backGroundColor[2]),
                            static_cast<uint8_t>(m_backGroundColor[3])};
        RGBA lineColor = {static_cast<uint8_t>(m_linesColor[0]),
                          static_cast<uint8_t>(m_linesColor[1]),
                          static_cast<uint8_t>(m_linesColor[2]),
                          static_cast<uint8_t>(m_linesColor[3])};
        for (int i = 0; i < m_iterations; ++i) {
            SDL_Surface* img =
                SurfacePainters::GenerateRGBSurface(colorNormal,
                                                    m_width,
                                                    m_height);
            for (int j = 0; j < m_normalLines; ++j) {
                std::random_device dev;
                std::mt19937 rng(dev());
                std::uniform_int_distribution<std::mt19937::result_type>
                    dist_width(
                    0, m_width - 1);
                std::uniform_int_distribution<std::mt19937::result_type>
                    dist_height(0, m_height - 1);
                SurfacePainters::DrawLine(img, lineColor,
                                          {dist_width(rng), dist_height(rng)},
                                          {dist_width(rng), dist_height(rng)});
            }

            for (int j = 0; j < m_bezierLines; ++j) {
                std::random_device dev;
                std::mt19937 rng(dev());
                std::uniform_int_distribution<std::mt19937::result_type>
                    dist_width(
                    0, m_width - 1);
                std::uniform_int_distribution<std::mt19937::result_type>
                    dist_height(0, m_height - 1);
                SurfacePainters::DrawCubicBezier(
                    img, lineColor, {dist_width(rng), dist_height(rng)},
                    {dist_width(rng), dist_height(rng)},
                    {dist_width(rng), dist_height(rng)},
                    {dist_width(rng), dist_height(rng)});
            }
            m_detector.setH(m_height);
            m_detector.setW(m_width);
            m_detector.setPixels((uint8_t*) img->pixels);
            m_detector.setStride(img->format->BytesPerPixel);
            auto detected = m_detector.Detect();

            auto detectedImg = SDL_CreateRGBSurface(
                0,
                img->w,
                img->h,
                img->format->BitsPerPixel,
                img->format->Rmask,
                img->format->Gmask,
                img->format->Bmask,
                img->format->Amask);

            std::copy(detected.get(),
                      detected.get()
                          + (img->w * img->h * img->format->BytesPerPixel),
                      (uint8_t*) detectedImg->pixels);
            m_allTimings.push_back(m_detector.GetTimings());

            if (!std::filesystem::exists(m_internalName)) {
                std::filesystem::create_directory(m_internalName);
            }

            if (!std::filesystem::exists(m_internalName + "/base")) {
                std::filesystem::create_directory(m_internalName + "/base");
            }

            std::string baseImgName =
                m_internalName + "/base/img_" + std::to_string(i) + ".png";
            IMG_SavePNG(img, baseImgName.c_str());

            if (!std::filesystem::exists(m_internalName + "/detected")) {
                std::filesystem::create_directory(m_internalName + "/detected");
            }

            std::string detectedImgName =
                m_internalName + "/detected/img_" + std::to_string(i) + ".png";
            IMG_SavePNG(detectedImg, detectedImgName.c_str());

            std::vector<float> distances;
            int misses = 0;

            for (int x = 0; x < detectedImg->w; ++x) {
                for (int y = 0; y < detectedImg->h; ++y) {
                    RGBA* color =
                        (RGBA*) (((uint8_t*) detectedImg->pixels) + (x * 4)
                            + (y * detectedImg->w * 4));
                    if (color->r != 0 && color->b != 0 && color->g != 0) {
                        SpiralIndexer indexer;
                        bool match = false;
                        for (int j = 0; j < 25; j++) {
                            int nX = x + indexer.X();
                            int nY = y + indexer.Y();
                            if (nX >= detectedImg->w || nY >= detectedImg->h) {
                                indexer++;
                                continue;
                            }

                            RGBA* color2 = (RGBA*) (((uint8_t*) img->pixels)
                                + (nX * 4) + (nY * img->w * 4));
                            if (color2->r == 255) {
                                float dis = DistanceOfPixels(x, y, nX, nY);
                                distances.push_back(dis);
                                match = true;
                                break;
                            }
                            indexer++;
                        }
                        if (!match) { misses++; }
                    }
                }
            }
            auto avg = std::reduce(distances.begin(), distances.end())
                / (float) distances.size();
            m_AVG.push_back(avg);
            m_missing.push_back(misses);

            SDL_FreeSurface(img);
            SDL_FreeSurface(detectedImg);

        }
    }
    void SaveData() override {
        if (m_allTimings.empty() || m_AVG.empty() || m_missing.empty()) {
            return;
        }
        if (!std::filesystem::exists(m_internalName)) {
            std::filesystem::create_directory(m_internalName);
        }
        std::ofstream out(m_internalName + "/" + m_internalName + ".txt");
        auto timingsIt = m_allTimings.begin();
        auto avgIt = m_AVG.begin();
        auto missingIt = m_missing.begin();
        out << "i miss avg t_all t_grey t_g t_blur t_sob t_max t_thresh t_hys"
            << std::endl;
        uint64_t counter = 1;
        for (; timingsIt != m_allTimings.end() && avgIt != m_AVG.end()
                   && missingIt != m_missing.end();
               ++timingsIt, ++avgIt, ++missingIt) {
            out << counter << " "
                << *avgIt << " "
                << *missingIt << " "
                << timingsIt->All_ms << " "
                << timingsIt->GrayScale_ms << " "
                << timingsIt->GaussCreation_ms << " "
                << timingsIt->Blur_ms << " "
                << timingsIt->SobelOperator_ms << " "
                << timingsIt->NonMaximumSuppression_ms << " "
                << timingsIt->DoubleThreshold_ms << " "
                << timingsIt->Hysteresis_ms << std::endl;
            counter++;
        }
        out.close();
    }

private:
    std::vector<CannyTimings> m_allTimings;
    T m_detector;
};

#endif //BSC_THESIS_TEST_INCLUDE_CANNY_CANNY_TESTER_H_
