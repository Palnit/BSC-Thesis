#include "Canny/Cpu/canny_cpu_tester.h"
#include <filesystem>
#include <random>
#include <thread>
#include "Canny/cpu/canny_edge_detector_cpu.h"
#include "SDL_image.h"
#include "general/cpu/gauss_blur_cpu.h"
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

        SDL_FreeSurface(img);
        SDL_FreeSurface(detected);
    }
}
CannyCpuTester::CannyCpuTester() : TesterBase("Canny Cpu Testing") {}
