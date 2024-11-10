#include "Canny/OpenCl/canny_edge_detector_open_cl.h"
#include "imgui.h"
#include "SDL_image.h"
#include "general/OpenCL/program.h"
#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/kernel.h"
#include "general/cpu/gauss_blur_cpu.h"
void CannyEdgeDetectorOpenCl::DisplayImGui() {

    if (ImGui::BeginTabItem(m_name.c_str())) {

        if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 21)) {
            if (m_gaussKernelSize % 2 == 0) {
                m_gaussKernelSize++;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        ImGui::SliderFloat("Standard Deviation",
                           &m_standardDeviation,
                           0.0001f,
                           30.0f);
        ImGui::SliderFloat("High Trash Hold",
                           &m_highTrashHold,
                           0.0f,
                           255.0f);
        ImGui::SliderFloat("Low Trash Hold",
                           &m_lowTrashHold,
                           0.0f,
                           255.0f);
        ImGui::Separator();
        if (ImGui::Button("Detect")) {
            DetectEdge();
        }
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
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "CannyTimings:");
        ImGui::Text("Whole execution:         %f ms", m_timings.All_ms);
        ImGui::Separator();
        ImGui::Text("Gray Scaling:            %f ms", m_timings.GrayScale_ms);
        ImGui::Text("Gauss Creation:          %f ms",
                    m_timings.GaussCreation_ms);
        ImGui::Text("Blur:                    %f ms", m_timings.Blur_ms);
        ImGui::Text("Sobel Operator:          %f ms",
                    m_timings.SobelOperator_ms);
        ImGui::Text("Non Maximum Suppression: %f ms",
                    m_timings.NonMaximumSuppression_ms);
        ImGui::Text("Double Threshold:        %f ms",
                    m_timings.DoubleThreshold_ms);
        ImGui::Text("Hysteresis:              %f ms", m_timings.Hysteresis_ms);

        ImGui::EndTabItem();
    }
}
void CannyEdgeDetectorOpenCl::DetectEdge() {

    ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);

    programTest.AddSource("OpenCLKernels/gauss_blur.cl");
    programTest.AddSource("OpenCLKernels/canny.cl");
    size_t size = (m_base->w * m_base->h * m_base->format->BytesPerPixel);

    ClWrapper::Memory<uint8_t, 0>
        image(programTest, (uint8_t*) m_base->pixels, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp2(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tangent(programTest, size, CL_MEM_READ_WRITE);
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
                                m_base->w,
                                CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> h(programTest,
                                m_base->h,
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
        m_base->w + (m_base->w % 32 != 0 ? (32 - m_base->w % 32) : 0);
    size_t
        height =
        m_base->h + (m_base->h % 32 != 0 ? (32 - m_base->h % 32) : 0);

    size_t missingW =
        (width / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t missingH =
        (height / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t widthNKernel = (m_base->w + missingW) % 32 != 0 ?
                          m_base->w + missingW
                              + (32 - (m_base->w + missingW) % 32) : m_base->w
                              + missingW;
    size_t heightNKernel = (m_base->h + missingH) % 32 != 0 ?
                           m_base->h + missingH
                               + (32 - (m_base->h + missingH) % 32) : m_base->h
                               + missingH;

    size_t missing3W =
        (width / 32) * (3 * (2 + 1));
    size_t missing3H =
        (height / 32) * (3 * (2 + 1));

    size_t width3Kernel = (m_base->w + missing3W) % 32 != 0 ?
                          m_base->w + missing3W
                              + (32 - (m_base->w + missing3W) % 32) : m_base->w
                              + missing3W;
    size_t height3Kernel = (m_base->h + missing3H) % 32 != 0 ?
                           m_base->h + missing3H
                               + (32 - (m_base->h + missing3H) % 32) : m_base->h
                               + missing3H;

    auto t1 = std::chrono::high_resolution_clock::now();

    m_timings.GrayScale_ms = ConvertToGreyScale(
        cl::NDRange(width, height),
        cl::NDRange(32, 32),
        image,
        tmp, w, h);

    m_timings.GaussCreation_ms = GetGaussian(
        cl::NDRange(m_gaussKernelSize,
                    m_gaussKernelSize),
        cl::NDRange(
            m_gaussKernelSize,
            m_gaussKernelSize),
        gauss,
        kernelSize,
        sigma);
    m_timings.Blur_ms = GaussianFilter(
        cl::NDRange(widthNKernel,
                    heightNKernel),
        cl::NDRange(32, 32),
        tmp,
        tmp2,
        gauss,
        kernelSize, w, h);

    m_timings.SobelOperator_ms = DetectionOperator(
        cl::NDRange(
            width3Kernel,
            height3Kernel),
        cl::NDRange(32, 32),
        tmp2,
        tmp,
        tangent,
        w,
        h);

    m_timings.NonMaximumSuppression_ms = NonMaximumSuppression(
        cl::NDRange(width3Kernel,
                    height3Kernel),
        cl::NDRange(32, 32),
        tmp,
        tmp2,
        tangent,
        w,
        h);

    m_timings.DoubleThreshold_ms = DoubleThreshold(
        cl::NDRange(width,
                    height),
        cl::NDRange(32,
                    32),
        tmp2,
        tmp,
        w,
        h, high, low);

    m_timings.Hysteresis_ms = Hysteresis(
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

    CopyBack(cl::NDRange(width, height),
             cl::NDRange(32, 32),
             tmp2,
             image, w, h);
    image.ReadFromDevice();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    m_timingsReady = true;
    std::copy(image.begin(), image.end(), (uint8_t*) m_detected->pixels);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_base->w,
                 m_base->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 image.begin());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}
void CannyEdgeDetectorOpenCl::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    VAO.UnBind();
    shaderProgram.UnBind();
}
