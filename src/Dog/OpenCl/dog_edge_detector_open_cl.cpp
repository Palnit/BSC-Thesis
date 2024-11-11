#include "Dog/OpenCl/dog_edge_detector_open_cl.h"
#include "SDL_image.h"
#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/kernel.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/program.h"
#include "general/cpu/gauss_blur_cpu.h"
#include "imgui.h"

void DogEdgeDetectorOpenCl::DetectEdge() {
    ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);
    programTest.AddSource("OpenCLKernels/gauss_blur.cl");
    programTest.AddSource("OpenCLKernels/dog.cl");

    size_t size = (m_base->w * m_base->h * m_base->format->BytesPerPixel);

    ClWrapper::Memory<uint8_t, 0> image(programTest, (uint8_t*) m_base->pixels,
                                        size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp2(programTest, size, CL_MEM_READ_WRITE);

    ClWrapper::Memory<float, 0> gauss1(
        programTest, m_gaussKernelSize * m_gaussKernelSize, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> gauss2(
        programTest, m_gaussKernelSize * m_gaussKernelSize, CL_MEM_READ_WRITE);

    ClWrapper::Memory<float, 0> finalGauss(
        programTest, m_gaussKernelSize * m_gaussKernelSize, CL_MEM_READ_WRITE);
    ClWrapper::Memory<int, 1> kernelSize(programTest, m_gaussKernelSize,
                                         CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 1> sigma1(programTest, m_standardDeviation1,
                                       CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 1> sigma2(programTest, m_standardDeviation2,
                                       CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> w(programTest, m_base->w, CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> h(programTest, m_base->h, CL_MEM_READ_WRITE);

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
    ClWrapper::Kernel DifferenceOfGaussian(programTest, "DifferenceOfGaussian");

    size_t width =
        m_base->w + (m_base->w % 32 != 0 ? (32 - m_base->w % 32) : 0);
    size_t height =
        m_base->h + (m_base->h % 32 != 0 ? (32 - m_base->h % 32) : 0);

    size_t missingW =
        (width / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t missingH =
        (height / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t widthNKernel = (m_base->w + missingW) % 32 != 0
        ? m_base->w + missingW + (32 - (m_base->w + missingW) % 32)
        : m_base->w + missingW;
    size_t heightNKernel = (m_base->h + missingH) % 32 != 0
        ? m_base->h + missingH + (32 - (m_base->h + missingH) % 32)
        : m_base->h + missingH;

    auto t1 = std::chrono::high_resolution_clock::now();

    m_timings.GrayScale_ms =
        Detectors::TimerRunner(ConvertToGreyScale, cl::NDRange(width, height),
                               cl::NDRange(32, 32), image, tmp, w, h);

    m_timings.Gauss1Creation_ms = Detectors::TimerRunner(
        GetGaussian, cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
        cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss1, kernelSize,
        sigma1);

    m_timings.Gauss2Creation_ms = Detectors::TimerRunner(
        GetGaussian, cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
        cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss2, kernelSize,
        sigma2);

    m_timings.DifferenceOfGaussian_ms = Detectors::TimerRunner(
        DifferenceOfGaussian, cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
        cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss1, gauss2,
        finalGauss, kernelSize);

    m_timings.Convolution_ms = Detectors::TimerRunner(
        GaussianFilter, cl::NDRange(widthNKernel, heightNKernel),
        cl::NDRange(32, 32), tmp, tmp2, finalGauss, kernelSize, w, h);

    CopyBack(cl::NDRange(width, height), cl::NDRange(32, 32), tmp2, image, w,
             h);
    image.ReadFromDevice();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    m_timingsReady = true;
    std::copy(image.begin(), image.end(), (uint8_t*) m_detected->pixels);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_base->w, m_base->h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, image.begin());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}
void DogEdgeDetectorOpenCl::DisplayImGui() {
    if (ImGui::BeginTabItem(m_name.c_str())) {
        if (OpenCLInfo::OPENCL_DEVICES[0]
                .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
            < 1024) {
            ImGui::Text("Not Enough Work Group");
            ImGui::EndTabItem();
            return;
        }

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
        if (ImGui::Button("Detect")) { DetectEdge(); }
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
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "DogTimings:");
        ImGui::Text("Whole execution:               %f ms", m_timings.All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:                  %f ms",
                    m_timings.GrayScale_ms);
        ImGui::Text("Gauss 1 Creation:              %f ms",
                    m_timings.Gauss1Creation_ms);
        ImGui::Text("Gauss 2 Creation:              %f ms",
                    m_timings.Gauss1Creation_ms);
        ImGui::Text("Difference of gaussian:        %f ms",
                    m_timings.DifferenceOfGaussian_ms);
        ImGui::Text("Convolution:                   %f ms",
                    m_timings.Convolution_ms);
        ImGui::EndTabItem();
    }
}
void DogEdgeDetectorOpenCl::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}
