//
// Created by Palnit on 2024. 01. 21.
//

#include "Dog/cuda/dog_edge_detector_cuda.cuh"
#include "general/cuda/gauss_blur.cuh"

__global__ void DifferenceOfGaussian(float* kernel1,
                                     float* kernel2,
                                     float* finalKernel,
                                     int kernelSize) {
    uint32_t x = threadIdx.x;
    uint32_t y = threadIdx.y;

    *(finalKernel + x + (y * kernelSize)) =
        *(kernel1 + x + (y * kernelSize)) - *(kernel2 + x + (y * kernelSize));

}

std::shared_ptr<uint8_t> CudaDogDetector::Detect() {

    m_detected =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) * m_w * m_h * m_stride));
    uint8_t* d_pixel = nullptr;

    cudaMalloc((void**) &d_pixel,
               sizeof(uint8_t) * m_w * m_h
                   * m_stride);

    cudaMemcpy(
        d_pixel, m_pixels,
        sizeof(uint8_t) * m_w * m_h * m_stride,
        cudaMemcpyHostToDevice);

    float* dest1;
    float* dest2;

    float* kernel1;
    float* kernel2;
    float* finalKernel;

    dim3 threads(32, 32);
    dim3 block
        (m_w / threads.x + (m_w % threads.x == 0 ? 0 : 1),
         m_h / threads.y
             + (m_h % threads.y == 0 ? 0 : 1));

    cudaMalloc((void**) &kernel1,
               sizeof(float) * m_gaussKernelSize * m_gaussKernelSize);
    cudaMalloc((void**) &kernel2,
               sizeof(float) * m_gaussKernelSize * m_gaussKernelSize);
    cudaMalloc((void**) &finalKernel,
               sizeof(float) * m_gaussKernelSize * m_gaussKernelSize);
    cudaMalloc((void**) &dest1, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &dest2, sizeof(float) * m_w * m_h);

    dim3 gauss(m_gaussKernelSize, m_gaussKernelSize);
    cudaEventRecord(m_timers.All_start);
    cudaEventRecord(m_timers.GrayScale_start);
    convertToGreyScale<<<block, threads>>>(d_pixel, dest1, m_w, m_h);
    cudaEventRecord(m_timers.GrayScale_stop);
    cudaEventSynchronize(m_timers.GrayScale_stop);

    cudaEventRecord(m_timers.Gauss1Creation_start);
    GetGaussian<<<1, gauss>>>(kernel1, m_gaussKernelSize, m_standardDeviation1);
    cudaEventRecord(m_timers.Gauss1Creation_stop);
    cudaEventSynchronize(m_timers.Gauss1Creation_stop);

    cudaEventRecord(m_timers.Gauss2Creation_start);
    GetGaussian<<<1, gauss>>>(kernel2, m_gaussKernelSize, m_standardDeviation2);
    cudaEventRecord(m_timers.Gauss2Creation_stop);
    cudaEventSynchronize(m_timers.Gauss2Creation_stop);

    cudaEventRecord(m_timers.DifferenceOfGaussian_start);
    DifferenceOfGaussian<<<1, gauss>>>(kernel1,
                                       kernel2,
                                       finalKernel,
                                       m_gaussKernelSize);
    cudaEventRecord(m_timers.DifferenceOfGaussian_stop);
    cudaEventSynchronize(m_timers.DifferenceOfGaussian_stop);

    cudaEventRecord(m_timers.Convolution_start);
    GaussianFilter<<<block, threads>>>(dest1,
                                       dest2,
                                       finalKernel,
                                       m_gaussKernelSize,
                                       m_w,
                                       m_h);
    cudaEventRecord(m_timers.Convolution_stop);
    cudaEventSynchronize(m_timers.Convolution_stop);

    CopyBack<<<block, threads>>>(d_pixel, dest2, m_w, m_h);
    cudaEventRecord(m_timers.All_stop);
    cudaEventSynchronize(m_timers.All_stop);

    cudaEventElapsedTime(&m_timings.All_ms,
                         m_timers.All_start,
                         m_timers.All_stop);
    cudaEventElapsedTime(&m_timings.GrayScale_ms,
                         m_timers.GrayScale_start,
                         m_timers.GrayScale_stop);
    cudaEventElapsedTime(&m_timings.Gauss1Creation_ms,
                         m_timers.Gauss1Creation_start,
                         m_timers.Gauss1Creation_stop);
    cudaEventElapsedTime(&m_timings.Gauss2Creation_ms,
                         m_timers.Gauss2Creation_start,
                         m_timers.Gauss2Creation_stop);
    cudaEventElapsedTime(&m_timings.DifferenceOfGaussian_ms,
                         m_timers.DifferenceOfGaussian_start,
                         m_timers.DifferenceOfGaussian_stop);
    cudaEventElapsedTime(&m_timings.Convolution_ms,
                         m_timers.Convolution_start,
                         m_timers.Convolution_stop);

    cudaMemcpy(m_detected, d_pixel,
               sizeof(uint8_t) * m_w * m_h * m_stride,
               cudaMemcpyDeviceToHost);

    cudaFree(dest1);
    cudaFree(dest2);
    cudaFree(kernel1);
    cudaFree(kernel2);
    cudaFree(finalKernel);
    cudaFree(d_pixel);
    cudaDeviceSynchronize();
    return std::shared_ptr<uint8_t>(m_detected);
}
