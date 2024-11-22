#include "Dog/cpu/dog_edge_detector_cpu.h"
#include <chrono>
#include "general/cpu/gauss_blur_cpu.h"

void DetectorsCPU::DifferenceOfGaussian(float* kernel1,
                                        float* kernel2,
                                        float* finalKernel,
                                        int kernelSize) {
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            *(finalKernel + i + (j * kernelSize)) =
                *(kernel1 + i + (j * kernelSize))
                    - *(kernel2 + i + (j * kernelSize));
        }
    }
}
std::shared_ptr<uint8_t> DogEdgeDetectorCPU::Detect() {
    m_detected =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) * m_w * m_h * m_stride));
    float* m_pixels1 = static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    float* m_pixels2 = static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    float* m_kernel1 = static_cast<float*>(
        malloc(sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    float* m_kernel2 = static_cast<float*>(
        malloc(sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    float* m_finalKernel = static_cast<float*>(
        malloc(sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));

    auto t1 = std::chrono::high_resolution_clock::now();

    m_timings.GrayScale_ms =
        Detectors::TimerRunner(DetectorsCPU::ConvertGrayScale,
                               (uint8_t*) m_pixels, m_pixels1, m_w, m_h);
    m_timings.Gauss1Creation_ms =
        Detectors::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel1,
                               m_gaussKernelSize, m_standardDeviation1);
    m_timings.Gauss2Creation_ms =
        Detectors::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel2,
                               m_gaussKernelSize, m_standardDeviation2);
    m_timings.DifferenceOfGaussian_ms =
        Detectors::TimerRunner(DetectorsCPU::DifferenceOfGaussian, m_kernel1,
                               m_kernel2, m_finalKernel, m_gaussKernelSize);

    m_timings.Convolution_ms = Detectors::TimerRunner(
        DetectorsCPU::GaussianFilter, m_pixels1, m_pixels2, m_finalKernel,
        m_gaussKernelSize, m_w, m_h);

    DetectorsCPU::CopyBack(m_detected, m_pixels2, m_w, m_h);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    free(m_pixels1);
    free(m_pixels2);
    free(m_kernel1);
    free(m_kernel2);
    free(m_finalKernel);
    return std::shared_ptr<uint8_t>(m_detected);
}
