#include <random>
#include <filesystem>
#include <numeric>
#include "Canny/OpenCl/canny_open_cl_tester.h"
#include "implot.h"
#include "SDL_surface.h"
#include "surface_painters.h"
#include "SDL_image.h"
#include "spiral_indexer.h"
#include "general/OpenCL/program.h"
#include "general/OpenCL/kernel.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/get_devices.h"
#include "general/cpu/gauss_blur_cpu.h"

void CannyOpenClTester::ResultDisplay() {
    if (m_selected) {
        auto* x = new float[m_iterations];
        auto* x2 = new int[m_iterations];
        for (int i = 0; i < m_iterations; i++) { x[i] = x2[i] = i; }
        if (ImGui::BeginTabBar("Errors")) {
            if (ImGui::BeginTabItem("Error Rate")) {
                if (ImPlot::BeginPlot("")) {
                    ImPlot::SetupAxes(
                        "Iteration",
                        "Avg closest true pixel",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     m_AVG.data(),
                                     m_AVG.size());
                    ImPlot::EndPlot();
                }
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Missing")) {
                if (ImPlot::BeginPlot("")) {
                    ImPlot::SetupAxes(
                        "Iteration",
                        "No real pixel found",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x2,
                                     m_missing.data(),
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
                        "Iteration",
                        "Time in ms",
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit,
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
                    ImPlot::PlotLine("",
                                     x,
                                     timing.data(),
                                     timing.size());
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
void CannyOpenClTester::SpecializedDisplayImGui() {
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
void CannyOpenClTester::Test() {
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
        int widthi = img->w;
        int heighti = img->h;

        for (int j = 0; j < m_normalLines; ++j) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist_width(
                0, widthi - 1);
            std::uniform_int_distribution<std::mt19937::result_type>
                dist_height(0, heighti - 1);
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
                0, widthi - 1);
            std::uniform_int_distribution<std::mt19937::result_type>
                dist_height(0, heighti - 1);
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

        ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);

        programTest.AddSource("OpenCLKernels/gauss_blur.cl");
        programTest.AddSource("OpenCLKernels/canny.cl");
        size_t size = (img->w * img->h * img->format->BytesPerPixel);

        ClWrapper::Memory<uint8_t, 0>
            image(programTest, (uint8_t*) img->pixels, size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0> tmp(programTest, size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0> tmp2(programTest, size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0>
            tangent(programTest, size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0>
            gauss(programTest,
                  m_gaussKernelSize * m_gaussKernelSize,
                  CL_MEM_READ_WRITE);
        ClWrapper::Memory<int, 1> kernelSize(programTest,
                                             m_gaussKernelSize,
                                             CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 1> sigma(programTest,
                                          m_standardDeviation,
                                          CL_MEM_READ_WRITE);

        ClWrapper::Memory<int, 1> w(programTest,
                                    img->w,
                                    CL_MEM_READ_WRITE);

        ClWrapper::Memory<int, 1> h(programTest,
                                    img->h,
                                    CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 1>
            high(programTest, m_highTrashHold, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 1>
            low(programTest, m_lowTrashHold, CL_MEM_READ_WRITE);

        image.WriteToDevice();
        kernelSize.WriteToDevice();
        sigma.WriteToDevice();
        w.WriteToDevice();
        h.WriteToDevice();
        high.WriteToDevice();
        low.WriteToDevice();
        ClWrapper::Kernel ConvertToGreyScale(programTest, "ConvertToGreyScale");
        ClWrapper::Kernel CopyBack(programTest, "CopyBack");
        ClWrapper::Kernel GetGaussian(programTest, "GetGaussian");
        ClWrapper::Kernel GaussianFilter(programTest, "GaussianFilter");
        ClWrapper::Kernel DetectionOperator(programTest, "DetectionOperator");
        ClWrapper::Kernel
            NonMaximumSuppression(programTest, "NonMaximumSuppression");

        ClWrapper::Kernel DoubleThreshold(programTest, "DoubleThreshold");
        ClWrapper::Kernel Hysteresis(programTest, "Hysteresis");
        size_t width =
            img->w + (img->w % 32 != 0 ? (32 - img->w % 32) : 0);
        size_t
            height =
            img->h + (img->h % 32 != 0 ? (32 - img->h % 32) : 0);

        size_t missingW =
            (width / 32)
                * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
        size_t missingH =
            (height / 32)
                * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
        size_t widthNKernel = (img->w + missingW) % 32 != 0 ?
                              img->w + missingW
                                  + (32 - (img->w + missingW) % 32) : img->w
                                  + missingW;
        size_t heightNKernel = (img->h + missingH) % 32 != 0 ?
                               img->h + missingH
                                   + (32 - (img->h + missingH) % 32) : img->h
                                   + missingH;

        size_t missing3W =
            (width / 32) * (3 * (2 + 1));
        size_t missing3H =
            (height / 32) * (3 * (2 + 1));

        size_t width3Kernel = (img->w + missing3W) % 32 != 0 ?
                              img->w + missing3W
                                  + (32 - (img->w + missing3W) % 32) : img->w
                                  + missing3W;
        size_t height3Kernel = (img->h + missing3H) % 32 != 0 ?
                               img->h + missing3H
                                   + (32 - (img->h + missing3H) % 32) : img->h
                                   + missing3H;

        auto t1 = std::chrono::high_resolution_clock::now();

        m_timings.GrayScale_ms =
            Detectors::TimerRunner(ConvertToGreyScale,
                                   cl::NDRange(width, height),
                                   cl::NDRange(32, 32),
                                   image,
                                   tmp, w, h);

        m_timings.GaussCreation_ms = Detectors::TimerRunner(GetGaussian,
                                                            cl::NDRange(
                                                                m_gaussKernelSize,
                                                                m_gaussKernelSize),
                                                            cl::NDRange(
                                                                m_gaussKernelSize,
                                                                m_gaussKernelSize),
                                                            gauss,
                                                            kernelSize,
                                                            sigma);
        m_timings.Blur_ms = Detectors::TimerRunner(GaussianFilter,
                                                   cl::NDRange(widthNKernel,
                                                               heightNKernel),
                                                   cl::NDRange(32, 32),
                                                   tmp,
                                                   tmp2,
                                                   gauss,
                                                   kernelSize, w, h);

        m_timings.SobelOperator_ms = Detectors::TimerRunner(DetectionOperator,
                                                            cl::NDRange(
                                                                width3Kernel,
                                                                height3Kernel),
                                                            cl::NDRange(32, 32),
                                                            tmp2,
                                                            tmp,
                                                            tangent,
                                                            w,
                                                            h);

        m_timings.NonMaximumSuppression_ms =
            Detectors::TimerRunner(NonMaximumSuppression,
                                   cl::NDRange(width3Kernel,
                                               height3Kernel),
                                   cl::NDRange(32, 32),
                                   tmp,
                                   tmp2,
                                   tangent,
                                   w,
                                   h);

        m_timings.DoubleThreshold_ms = Detectors::TimerRunner(DoubleThreshold,
                                                              cl::NDRange(width,
                                                                          height),
                                                              cl::NDRange(32,
                                                                          32),
                                                              tmp2,
                                                              tmp,
                                                              w,
                                                              h, high, low);

        m_timings.Hysteresis_ms = Detectors::TimerRunner(Hysteresis,
                                                         cl::NDRange(
                                                             width3Kernel,
                                                             height3Kernel),
                                                         cl::NDRange(32, 32),
                                                         tmp,
                                                         tmp2,
                                                         w,
                                                         h);

        CopyBack(cl::NDRange(width, height),
                 cl::NDRange(32, 32),
                 tmp2,
                 image, w, h);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> time = t2 - t1;
        image.ReadFromDevice();
        m_timings.All_ms = time.count();
        m_allTimings.push_back(m_timings);

        auto detected = SDL_CreateRGBSurface(
            0, img->w, img->h, img->format->BitsPerPixel, img->format->Rmask,
            img->format->Gmask, img->format->Bmask, img->format->Amask);

        std::copy(image.begin(), image.end(), (uint8_t*) detected->pixels);

        if (!std::filesystem::exists("./open_cl_canny")) {
            std::filesystem::create_directory("./open_cl_canny");
        }

        if (!std::filesystem::exists("./open_cl_canny/base")) {
            std::filesystem::create_directory("./open_cl_canny/base");
        }
        std::string baseImgName = "./open_cl_canny/base/img_";
        baseImgName += std::to_string(i) + ".png";
        IMG_SavePNG(img, baseImgName.c_str());

        if (!std::filesystem::exists("./open_cl_canny/detected")) {
            std::filesystem::create_directory("./open_cl_canny/detected");
        }
        std::string detectedImgName = "./open_cl_canny/detected/img_";
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
CannyOpenClTester::CannyOpenClTester() : TesterBase("Canny OpenCl Tester") {}
