#include "Dog/OpenCl/dog_open_cl_tester.h"
#include <SDL_image.h>
#include <SDL_surface.h>
#include <implot.h>
#include <filesystem>
#include <numeric>
#include <random>
#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/kernel.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/program.h"
#include "general/cpu/gauss_blur_cpu.h"
#include "spiral_indexer.h"
#include "surface_painters.h"

DogOpenClTester::DogOpenClTester() : TesterBase("Dog OpenCl Tester") {}

void DogOpenClTester::ResultDisplay() {
    if (m_selected) {
        if (OpenCLInfo::OPENCL_DEVICES[0]
                .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
            < 1024) {
            ImGui::Text("Not Enough Work Group");
            return;
        }
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

            if (ImGui::BeginTabItem("Gauss 1 Creation")) {
                std::vector<float> timing;
                for (const auto& allTiming : m_allTimings) {
                    timing.emplace_back(allTiming.Gauss1Creation_ms);
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

            if (ImGui::BeginTabItem("Gauss 2 Creation")) {
                std::vector<float> timing;
                for (const auto& allTiming : m_allTimings) {
                    timing.emplace_back(allTiming.Gauss2Creation_ms);
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

            if (ImGui::BeginTabItem("Difference Of Gaussian")) {
                std::vector<float> timing;
                for (const auto& allTiming : m_allTimings) {
                    timing.emplace_back(allTiming.DifferenceOfGaussian_ms);
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

            if (ImGui::BeginTabItem("Convolution")) {
                std::vector<float> timing;
                for (const auto& allTiming : m_allTimings) {
                    timing.emplace_back(allTiming.Convolution_ms);
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
void DogOpenClTester::SpecializedDisplayImGui() {
    if (OpenCLInfo::OPENCL_DEVICES[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
        < 1024) {
        ImGui::Text("Not Enough Work Group");
        return;
    }

    ImGui::SeparatorText("Dog Settings");

    if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 21)) {
        if (m_gaussKernelSize % 2 == 0) { m_gaussKernelSize++; }
    }
    ImGui::SetItemTooltip("Only Odd Numbers");
    if (ImGui::SliderFloat("Standard Deviation 1", &m_standardDeviation1,
                           0.0001f, 30.0f)) {
        if (m_standardDeviation1 >= m_standardDeviation2) {
            m_standardDeviation1--;
        }
    }
    ImGui::SetItemTooltip("Standard Deviation 1 should be smaller than 2");
    if (ImGui::SliderFloat("Standard Deviation 2", &m_standardDeviation2,
                           0.0001f, 30.0f)) {
        if (m_standardDeviation1 >= m_standardDeviation2) {
            m_standardDeviation2++;
        }
    }
    ImGui::SliderFloat("Simple Threshold", &m_threshold, 0.f, 255.f);
}

void DogOpenClTester::Test() {
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
            SurfacePainters::GenerateRGBSurface(colorNormal, m_width, m_height);
        int widthi = img->w;
        int heighti = img->h;

        for (int j = 0; j < m_normalLines; ++j) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist_width(
                0, widthi - 1);
            std::uniform_int_distribution<std::mt19937::result_type>
                dist_height(0, heighti - 1);
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
            SurfacePainters::DrawCubicBezier(
                img, lineColor, {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)},
                {dist_width(rng), dist_height(rng)});
        }

        ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);
        programTest.AddSource("OpenCLKernels/gauss_blur.cl");
        programTest.AddSource("OpenCLKernels/dog.cl");

        size_t size = (img->w * img->h * img->format->BytesPerPixel);

        ClWrapper::Memory<uint8_t, 0> image(programTest, (uint8_t*) img->pixels,
                                            size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0> tmp(programTest, size, CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0> tmp2(programTest, size, CL_MEM_READ_WRITE);

        ClWrapper::Memory<float, 0> gauss1(
            programTest, m_gaussKernelSize * m_gaussKernelSize,
            CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 0> gauss2(
            programTest, m_gaussKernelSize * m_gaussKernelSize,
            CL_MEM_READ_WRITE);

        ClWrapper::Memory<float, 0> finalGauss(
            programTest, m_gaussKernelSize * m_gaussKernelSize,
            CL_MEM_READ_WRITE);
        ClWrapper::Memory<int, 1> kernelSize(programTest, m_gaussKernelSize,
                                             CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 1> sigma1(programTest, m_standardDeviation1,
                                           CL_MEM_READ_WRITE);
        ClWrapper::Memory<float, 1> sigma2(programTest, m_standardDeviation2,
                                           CL_MEM_READ_WRITE);

        ClWrapper::Memory<int, 1> w(programTest, img->w, CL_MEM_READ_WRITE);

        ClWrapper::Memory<int, 1> h(programTest, img->h, CL_MEM_READ_WRITE);

        image.WriteToDevice();
        kernelSize.WriteToDevice();
        sigma1.WriteToDevice();
        sigma2.WriteToDevice();
        w.WriteToDevice();
        h.WriteToDevice();

        ClWrapper::Kernel ConvertToGreyScale(programTest, "ConvertToGreyScale");
        ClWrapper::Kernel CopyBack(programTest, "CopyBack");
        ClWrapper::Kernel GetGaussian(programTest, "GetGaussian");
        ClWrapper::Kernel GaussianFilter(programTest, "GaussianFilter");
        ClWrapper::Kernel DifferenceOfGaussian(programTest,
                                               "DifferenceOfGaussian");

        size_t width = img->w + (img->w % 32 != 0 ? (32 - img->w % 32) : 0);
        size_t height = img->h + (img->h % 32 != 0 ? (32 - img->h % 32) : 0);

        size_t missingW = (width / 32)
            * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
        size_t missingH = (height / 32)
            * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
        size_t widthNKernel = (img->w + missingW) % 32 != 0
            ? img->w + missingW + (32 - (img->w + missingW) % 32)
            : img->w + missingW;
        size_t heightNKernel = (img->h + missingH) % 32 != 0
            ? img->h + missingH + (32 - (img->h + missingH) % 32)
            : img->h + missingH;

        auto t1 = std::chrono::high_resolution_clock::now();

        m_timings.GrayScale_ms = Detectors::TimerRunner(
            ConvertToGreyScale, cl::NDRange(width, height), cl::NDRange(32, 32),
            image, tmp, w, h);

        m_timings.Gauss1Creation_ms = Detectors::TimerRunner(
            GetGaussian, cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
            cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss1,
            kernelSize, sigma1);

        m_timings.Gauss2Creation_ms = Detectors::TimerRunner(
            GetGaussian, cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
            cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss2,
            kernelSize, sigma2);

        m_timings.DifferenceOfGaussian_ms = Detectors::TimerRunner(
            DifferenceOfGaussian,
            cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
            cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss1, gauss2,
            finalGauss, kernelSize);

        m_timings.Convolution_ms = Detectors::TimerRunner(
            GaussianFilter, cl::NDRange(widthNKernel, heightNKernel),
            cl::NDRange(32, 32), tmp, tmp2, finalGauss, kernelSize, w, h);

        CopyBack(cl::NDRange(width, height), cl::NDRange(32, 32), tmp2, image,
                 w, h);
        image.ReadFromDevice();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> time = t2 - t1;
        m_timings.All_ms = time.count();

        auto detected = SDL_CreateRGBSurface(
            0, img->w, img->h, img->format->BitsPerPixel, img->format->Rmask,
            img->format->Gmask, img->format->Bmask, img->format->Amask);

        std::copy(image.begin(), image.end(), (uint8_t*) detected->pixels);
        m_allTimings.push_back(m_timings);

        if (!std::filesystem::exists("./open_cl_dog")) {
            std::filesystem::create_directory("./open_cl_dog");
        }

        if (!std::filesystem::exists("./open_cl_dog/base")) {
            std::filesystem::create_directory("./open_cl_dog/base");
        }
        std::string baseImgName = "./open_cl_dog/base/img_";
        baseImgName += std::to_string(i) + ".png";
        IMG_SavePNG(img, baseImgName.c_str());

        if (!std::filesystem::exists("./open_cl_dog/detected")) {
            std::filesystem::create_directory("./open_cl_dog/detected");
        }
        std::string detectedImgName = "./open_cl_dog/detected/img_";
        detectedImgName += std::to_string(i) + ".png";
        IMG_SavePNG(detected, detectedImgName.c_str());
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
                        if (nX >= detected->w || nY >= detected->h) {
                            indexer++;
                            continue;
                        }

                        RGBA* color2 =
                            (RGBA*) (((uint8_t*) detected->pixels) + (nX * 4)
                                     + (nY * detected->w * 4));
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
        SDL_FreeSurface(detected);
    }
}
