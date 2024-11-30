#ifndef BSC_THESIS_TEST_INCLUDE_DOG_DOG_TESTER_H_
#define BSC_THESIS_TEST_INCLUDE_DOG_DOG_TESTER_H_

#include <random>
#include <numeric>
#include <fstream>
#include <iostream>
#include "tester_base.h"
#include "Dog/dog_timings.h"
#include "Dog/dog_edge_detector.h"
#include "implot.h"
#include "Dog/OpenCl/dog_edge_detector_open_cl.h"
#include "general/OpenCL/get_devices.h"
#include "SDL_surface.h"
#include "surface_painters.h"
#include "SDL_image.h"
#include "spiral_indexer.h"

template<class T>
class DogTester : public TesterBase {
public:
    static_assert(std::is_base_of<DogEdgeDetector, T>::value,
                  "Template type must have a base type of DogEdgeDetector");

    explicit DogTester(const std::string& name,
                       const std::string& internalName)
        : TesterBase(name, internalName), m_threshold(5) {}

    void ResultDisplay() override {
        if (m_selected) {
            if (std::is_same_v<DogEdgeDetectorOpenCl, T>) {
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

                if (ImGui::BeginTabItem("Gauss 1 Creation")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.Gauss1Creation_ms);
                    }
                    if (ImPlot::BeginPlot("Gauss 1 Creation")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Gauss 2 Creation")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.Gauss2Creation_ms);
                    }
                    if (ImPlot::BeginPlot("Gauss 2 Creation")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Difference Of Gaussian")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.DifferenceOfGaussian_ms);
                    }
                    if (ImPlot::BeginPlot("Difference Of Gaussian")) {
                        ImPlot::SetupAxes(
                            "Iteration", "Time in ms",
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                            ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                        ImPlot::PlotLine("", x, timing.data(), timing.size());
                        ImPlot::EndPlot();
                    }
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Convolution")) {
                    std::vector<float> timing;
                    for (const auto& allTiming : m_allTimings) {
                        timing.emplace_back(allTiming.Convolution_ms);
                    }
                    if (ImPlot::BeginPlot("Convolution")) {
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
        if (std::is_same_v<DogEdgeDetectorOpenCl, T>) {
            if (OpenCLInfo::OPENCL_DEVICES[0]
                .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                < 1024) {
                ImGui::Text("Not Enough Work Group");
                return;
            }
        }
        ImGui::SeparatorText("DoG Settings");

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
        ImGui::SliderFloat("Simple Threshold", &m_threshold, 0.f, 255.f);
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

            for (int x = 0; x < img->w; ++x) {
                for (int y = 0; y < img->h; ++y) {
                    RGBA* color = (RGBA*) (((uint8_t*) img->pixels) + (x * 4)
                        + (y * img->w * 4));
                    if (lineColor == *color) {
                        SpiralIndexer indexer;
                        bool match = false;
                        for (int j = 0; j < 25; j++) {
                            int nX = x + indexer.X();
                            int nY = y + indexer.Y();
                            if (nX >= detectedImg->w || nY >= detectedImg->h) {
                                indexer++;
                                continue;
                            }

                            RGBA* color2 =
                                (RGBA*) (((uint8_t*) detectedImg->pixels)
                                    + (nX * 4)
                                    + (nY * detectedImg->w * 4));
                            if (color2->r <= m_threshold) {
                                indexer++;
                                continue;
                            }
                            float dis = DistanceOfPixels(x, y, nX, nY);
                            distances.push_back(dis);
                            match = true;
                            break;
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
        out << "i miss avg t_all t_grey t_g1 t_g2 t_diff t_conv" << std::endl;
        uint64_t counter = 1;
        for (; timingsIt != m_allTimings.end() && avgIt != m_AVG.end()
                   && missingIt != m_missing.end();
               ++timingsIt, ++avgIt, ++missingIt) {
            out << counter << " "
                << *avgIt << " "
                << *missingIt << " "
                << timingsIt->All_ms << " "
                << timingsIt->GrayScale_ms << " "
                << timingsIt->Gauss1Creation_ms << " "
                << timingsIt->Gauss2Creation_ms << " "
                << timingsIt->DifferenceOfGaussian_ms << " "
                << timingsIt->Convolution_ms << std::endl;
            counter++;
        }
        out.close();
    }
private:
    std::vector<DogTimings> m_allTimings;
    float m_threshold;
    T m_detector;

};

#endif //BSC_THESIS_TEST_INCLUDE_DOG_DOG_TESTER_H_
