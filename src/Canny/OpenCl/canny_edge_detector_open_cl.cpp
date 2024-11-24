#include "Canny/OpenCl/canny_edge_detector_open_cl.h"
#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/kernel.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/program.h"

std::shared_ptr<uint8_t> CannyEdgeDetectorOpenCl::Detect() {
    m_detected =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) * m_w * m_h * m_stride));
    ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);

    programTest.AddSource("OpenCLKernels/gauss_blur.cl");
    programTest.AddSource("OpenCLKernels/canny.cl");
    size_t size = (m_w * m_h * m_stride);

    ClWrapper::Memory<uint8_t, 0> image(programTest, m_pixels, size,
                                        CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tmp2(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> tangent(programTest, size, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 0> gauss(
        programTest, m_gaussKernelSize * m_gaussKernelSize, CL_MEM_READ_WRITE);
    ClWrapper::Memory<int, 1> kernelSize(programTest, m_gaussKernelSize,
                                         CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 1> sigma(programTest, m_standardDeviation,
                                      CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> w(programTest, m_w, CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> h(programTest, m_h, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 1> high(programTest, m_high, CL_MEM_READ_WRITE);
    ClWrapper::Memory<float, 1> low(programTest, m_low, CL_MEM_READ_WRITE);

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
    ClWrapper::Kernel NonMaximumSuppression(programTest,
                                            "NonMaximumSuppression");

    ClWrapper::Kernel DoubleThreshold(programTest, "DoubleThreshold");
    ClWrapper::Kernel Hysteresis(programTest, "Hysteresis");
    size_t width = m_w + (m_w % 32 != 0 ? (32 - m_w % 32) : 0);
    size_t height = m_h + (m_h % 32 != 0 ? (32 - m_h % 32) : 0);

    auto t1 = std::chrono::high_resolution_clock::now();

    m_timings.GrayScale_ms = ConvertToGreyScale(
        cl::NDRange(width, height), cl::NDRange(32, 32), image, tmp, w, h);

    m_timings.GaussCreation_ms =
        GetGaussian(cl::NDRange(m_gaussKernelSize, m_gaussKernelSize),
                    cl::NDRange(m_gaussKernelSize, m_gaussKernelSize), gauss,
                    kernelSize, sigma);
    m_timings.Blur_ms =
        GaussianFilter(cl::NDRange(width, height),
                       cl::NDRange(32, 32), tmp, tmp2, gauss, kernelSize, w, h);

    m_timings.SobelOperator_ms =
        DetectionOperator(cl::NDRange(width, height),
                          cl::NDRange(32, 32), tmp2, tmp, tangent, w, h);

    m_timings.NonMaximumSuppression_ms =
        NonMaximumSuppression(cl::NDRange(width, height),
                              cl::NDRange(32, 32), tmp, tmp2, tangent, w, h);

    m_timings.DoubleThreshold_ms =
        DoubleThreshold(cl::NDRange(width, height), cl::NDRange(32, 32), tmp2,
                        tmp, w, h, high, low);

    m_timings.Hysteresis_ms =
        Hysteresis(cl::NDRange(width, height),
                   cl::NDRange(32, 32), tmp, tmp2, w, h);

    CopyBack(cl::NDRange(width, height), cl::NDRange(32, 32), tmp2, image, w,
             h);
    image.ReadFromDevice();

    std::copy(image.begin(), image.begin() + size, m_detected);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    return std::shared_ptr<uint8_t>(m_detected);
}
