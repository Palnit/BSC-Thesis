#include "Canny/Cpu/canny_cpu_tester.h"
#include <imgui.h>
#include <implot.h>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>
#include "Canny/cpu/canny_edge_detector_cpu.h"
#include "SDL_image.h"
#include "general/cpu/gauss_blur_cpu.h"
#include "spiral_indexer.h"
#include "surface_painters.h"

void CannyCpuTester::SpecializedDisplayImGui() {
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

void CannyCpuTester::Test() {
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

        m_pixels1 = static_cast<float*>(malloc(sizeof(float) * width * height));
        m_pixels2 = static_cast<float*>(malloc(sizeof(float) * width * height));
        m_kernel = static_cast<float*>(
            malloc(sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
        m_tangent = static_cast<float*>(malloc(sizeof(float) * width * height));

        auto t1 = std::chrono::high_resolution_clock::now();
        m_timings.GrayScale_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::ConvertGrayScale, (uint8_t*) img->pixels, m_pixels1,
            width, height);
        m_timings.GaussCreation_ms =
            DetectorsCPU::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel,
                                      m_gaussKernelSize, m_standardDeviation);

        m_timings.Blur_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::GaussianFilter, m_pixels1, m_pixels2, m_kernel,
            m_gaussKernelSize, width, height);
        m_timings.SobelOperator_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::DetectionOperator, m_pixels2, m_pixels1, m_tangent,
            width, height);
        m_timings.NonMaximumSuppression_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::NonMaximumSuppression, m_pixels1, m_pixels2,
            m_pixels2, width, height);
        m_timings.DoubleThreshold_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::DoubleThreshold, m_pixels2, m_pixels1, width, height,
            m_highTrashHold, m_lowTrashHold);
        m_timings.Hysteresis_ms = DetectorsCPU::TimerRunner(
            DetectorsCPU::Hysteresis, m_pixels1, m_pixels2, width, height);

        auto detected = SDL_CreateRGBSurface(
            0, img->w, img->h, img->format->BitsPerPixel, img->format->Rmask,
            img->format->Gmask, img->format->Bmask, img->format->Amask);

        DetectorsCPU::CopyBack((uint8_t*) detected->pixels, m_pixels2, width,
                               height);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> time = t2 - t1;
        m_timings.All_ms = time.count();
        m_allTimings.push_back(m_timings);

        if (!std::filesystem::exists("./base")) {
            std::filesystem::create_directory("./base");
        }
        std::string baseImgName = "./base/img_";
        baseImgName += std::to_string(i) + ".png";
        IMG_SavePNG(img, baseImgName.c_str());

        if (!std::filesystem::exists("./detected")) {
            std::filesystem::create_directory("./detected");
        }
        std::string detectedImgName = "./detected/img_";
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
                    for (int i = 0; i < 25; i++) {
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
        std::cout << "AVG: " << avg << std::endl;
        std::cout << "Missing: " << misses << std::endl;

        SDL_FreeSurface(img);
        SDL_FreeSurface(detected);
    }
}
CannyCpuTester::CannyCpuTester() : TesterBase("Canny Cpu Testing") {}

void CannyCpuTester::ResultDisplay() {
    float x[m_iterations];
    for (int i = 0; i < m_iterations; i++) { x[i] = i; }
    if (ImGui::BeginTabItem(m_name.c_str())) {
        if (ImPlot::BeginPlot("Error rate")) {
            ImPlot::SetupAxes(
                "x", "y", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
            ImPlot::PlotLine("Avg's", x, m_AVG.data(), m_AVG.size());
            ImPlot::EndPlot();
        }
        ImGui::EndTabItem();
    }
}