#include "Canny/Cuda/canny_cuda_tester.h"
#include <cuda_runtime.h>
#include <imgui.h>
#include <implot.h>
#include <filesystem>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include "Canny/cpu/canny_detector_cpu.h"
#include "Canny/cuda/cuda_canny_edge_detection.cuh"
#include "SDL_image.h"
#include "general/cpu/gauss_blur_cpu.h"
#include "spiral_indexer.h"
#include "surface_painters.h"

CannyCudaTester::CannyCudaTester() : TesterBase("Canny Cuda Tester") {}
void CannyCudaTester::SpecializedDisplayImGui() {
    ImGui::SeparatorText("Canny Settings");

    if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 21)) {
        if (m_gaussKernelSize % 2 == 0) { m_gaussKernelSize++; }
    }
    ImGui::SetItemTooltip("Only Odd Numbers");
    ImGui::SliderFloat("Standard Deviation", &m_standardDeviation, 0.0001f,
                       30.0f);
    ImGui::SliderFloat("High Trash Hold", &m_highTrashHold, 0.0f, 255.0f);
    ImGui::SliderFloat("Low Trash Hold", &m_lowTrashHold, 0.0f, 255.0f);
}
void CannyCudaTester::ResultDisplay() {
    if (m_selected) {
        auto* x = new float[m_iterations];
        auto* x2 = new int[m_iterations];
        for (int i = 0; i < m_iterations; i++) { x[i] = x2[i] = i; }
        if (ImGui::BeginTabBar("Errors")) {
            if (ImGui::BeginTabItem("Error Rate")) {
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
                if (ImPlot::BeginPlot("")) {
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
        delete[] x;
        delete[] x2;
    }
}
void CannyCudaTester::Test() {
    m_AVG.clear();
    m_allTimings.clear();
    m_missing.clear();

    for (int i = 0; i < m_iterations; ++i) {
        RGBA color = {static_cast<uint8_t>(m_backGroundColor[0]),
                      static_cast<uint8_t>(m_backGroundColor[1]),
                      static_cast<uint8_t>(m_backGroundColor[2]),
                      static_cast<uint8_t>(m_backGroundColor[3])};
        SDL_Surface* img =
            SurfacePainters::GenerateRGBSurface(color, m_width, m_height);
        int width = img->w;
        int height = img->h;

        for (int j = 0; j < m_normalLines; ++j) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist_width(
                0, width - 1);
            std::uniform_int_distribution<std::mt19937::result_type>
                dist_height(0, height - 1);
            RGBA lineColor = {static_cast<uint8_t>(m_linesColor[0]),
                              static_cast<uint8_t>(m_linesColor[1]),
                              static_cast<uint8_t>(m_linesColor[2]),
                              static_cast<uint8_t>(m_linesColor[3])};
            SurfacePainters::DrawLine(img, lineColor,
                                      {dist_width(rng), dist_height(rng)},
                                      {dist_width(rng), dist_height(rng)});
        }

        for (int j = 0; j < m_bezierLines; ++j) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist_width(
                0, width - 1);
            std::uniform_int_distribution<std::mt19937::result_type>
                dist_height(0, height - 1);
            RGBA lineColor = {static_cast<uint8_t>(m_linesColor[0]),
                              static_cast<uint8_t>(m_linesColor[1]),
                              static_cast<uint8_t>(m_linesColor[2]),
                              static_cast<uint8_t>(m_linesColor[3])};
            SurfacePainters::DrawCubicBezier(
                img, lineColor, {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)});
        }

        uint8_t* d_pixel = nullptr;

        cudaMalloc(
            (void**) &d_pixel,
            sizeof(uint8_t) * img->w * img->h * img->format->BytesPerPixel);

        cudaMemcpy(
            d_pixel, img->pixels,
            sizeof(uint8_t) * img->w * img->h * img->format->BytesPerPixel,
            cudaMemcpyHostToDevice);

        CudaCannyDetector detector(d_pixel, img->w, img->h, m_gaussKernelSize,
                                   m_standardDeviation, m_highTrashHold,
                                   m_lowTrashHold);
        m_timings = detector.GetTimings();
        m_allTimings.push_back(m_timings);

        auto detected = SDL_CreateRGBSurface(
            0, img->w, img->h, img->format->BitsPerPixel, img->format->Rmask,
            img->format->Gmask, img->format->Bmask, img->format->Amask);

        cudaMemcpy(detected->pixels, d_pixel,
                   sizeof(uint8_t) * detected->w * detected->h
                       * detected->format->BytesPerPixel,
                   cudaMemcpyDeviceToHost);

        cudaFree(d_pixel);

        if (!std::filesystem::exists("./cuda_canny")) {
            std::filesystem::create_directory("./cuda_canny");
        }

        if (!std::filesystem::exists("./cuda_canny/base")) {
            std::filesystem::create_directory("./cuda_canny/base");
        }
        std::string baseImgName = "./cuda_canny/base/img_";
        baseImgName += std::to_string(i) + ".png";
        IMG_SavePNG(img, baseImgName.c_str());

        if (!std::filesystem::exists("./cuda_canny/detected")) {
            std::filesystem::create_directory("./cuda_canny/detected");
        }
        std::string detectedImgName = "./cuda_canny/detected/img_";
        detectedImgName += std::to_string(i) + ".png";
        IMG_SavePNG(detected, detectedImgName.c_str());
        std::vector<float> distances;
        int misses = 0;

        for (int x = 0; x < detected->w; ++x) {
            for (int y = 0; y < detected->h; ++y) {
                RGBA* color = (RGBA*) (((uint8_t*) detected->pixels) + (x * 4)
                                       + (y * detected->w * 4));
                if (color->r != 0 && color->b != 0 && color->g != 0) {
                    SpiralIndexer indexer;
                    bool match = false;
                    for (int j = 0; j < 25; j++) {
                        int nX = x + indexer.X();
                        int nY = y + indexer.Y();
                        if (nX >= detected->w || nY >= detected->h) {
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
        SDL_FreeSurface(detected);
    }
}
