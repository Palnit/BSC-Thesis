#include "Dog/OpenCl/dog_edge_detector_open_cl.h"
#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/kernel.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/program.h"
#include "general/cpu/gauss_blur_cpu.h"
std::shared_ptr<uint8_t> DogEdgeDetectorOpenCl::Detect() {
    ClWrapper::Program programTest(OpenCLInfo::OPENCL_DEVICES[0]);
    programTest.AddSource("OpenCLKernels/gauss_blur.cl");
    programTest.AddSource("OpenCLKernels/dog.cl");

    size_t size = (m_w * m_h * m_stride);

    ClWrapper::Memory<uint8_t, 0> image(programTest, (uint8_t*) m_pixels, size,
                                        CL_MEM_READ_WRITE);
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

    ClWrapper::Memory<int, 1> w(programTest, m_w, CL_MEM_READ_WRITE);

    ClWrapper::Memory<int, 1> h(programTest, m_h, CL_MEM_READ_WRITE);

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

    size_t width = m_w + (m_w % 32 != 0 ? (32 - m_w % 32) : 0);
    size_t height = m_h + (m_h % 32 != 0 ? (32 - m_h % 32) : 0);

    size_t missingW =
        (width / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t missingH =
        (height / 32) * (m_gaussKernelSize * 2 + (m_gaussKernelSize - 1 / 2));
    size_t widthNKernel = (m_w + missingW) % 32 != 0
        ? m_w + missingW + (32 - (m_w + missingW) % 32)
        : m_w + missingW;
    size_t heightNKernel = (m_h + missingH) % 32 != 0
        ? m_h + missingH + (32 - (m_h + missingH) % 32)
        : m_h + missingH;

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
    return std::shared_ptr<uint8_t>(m_detected);
}
