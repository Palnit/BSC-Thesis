//
// Created by Palnit on 2023. 11. 12.
//

#include "Canny/cuda/canny_edge_detector_cuda.cuh"
#include <cstdio>
#include <math_constants.h>
#include "general/cuda/gauss_blur.cuh"

__global__ void DetectionOperator(float* src,
                                  float* dest,
                                  float* tangent,
                                  int w,
                                  int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};

    float SumX = 0;
    float SumY = 0;

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int ix = x + i;
            int jx = y + j;

            if (ix < 0) { ix = 0; }
            if (ix >= w) { ix = w - 1; }
            if (jx < 0) { jx = 0; }
            if (jx >= h) { jx = h - 1; }
            SumX = std::fmaf(*(src + ix + (jx * w)),
                             *(SobelX + (i + 1) + ((j + 1) * 3)), SumX);
            SumY = std::fmaf(*(src + ix + (jx * w)),
                             *(SobelY + (i + 1) + ((j + 1) * 3)), SumY);
        }
    }
    *(dest + x + (y * w)) = hypotf(SumX, SumY);
    float angle = (atan2(SumX, SumY) * 180.f) / CUDART_PI_F;
    if (angle < 0) {
        angle += 180;
    }
    *(tangent + x + (y * w)) = angle;

}

__global__ void NonMaximumSuppression(float* src,
                                      float* dest,
                                      float* tangent,
                                      int w,
                                      int h) {

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    int yp;
    int yn;
    int xp;
    int xn;

    float* tangentA = (tangent + x + (y * w));
    float gradientA = *(src + x + (y * w));
    float gradientP = 2000;
    float gradientN = 2000;

    yp = y + 1;
    if (yp >= h) { yp = h - 1; }
    xp = x + 1;
    if (xp >= h) { xp = h - 1; }
    yn = y - 1;
    if (yn < 0) { yn = 0; }
    xn = x - 1;
    if (xn < 0) { xn = 0; }

    if ((0 <= *tangentA && *tangentA < 22.5)
        || (157.5 <= *tangentA && *tangentA <= 180)) {
        gradientP = *(src + x + (yp * w));
        gradientN = *(src + x + (yn * w));
    } else if (22.5 <= *tangentA && *tangentA < 67.5) {
        gradientP = *(src + xp + (yn * w));
        gradientN = *(src + xn + (yp * w));
    } else if (67.5 <= *tangentA && *tangentA < 112.5) {
        gradientP = *(src + xp + (y * w));
        gradientN = *(src + xn + (y * w));
    } else if (112.5 <= *tangentA && *tangentA < 157.5) {
        gradientP = *(src + xn + (yn * w));
        gradientN = *(src + xp + (yp * w));
    }

    if (gradientA < gradientN || gradientA < gradientP) {
        gradientA = 0.f;
    }
    *(dest + x + (y * w)) = gradientA;
}

__global__ void DoubleThreshold(float* gradient_in,
                                float* gradient_out,
                                int w,
                                int h,
                                float high,
                                float low) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    if (*(gradient_in + x + (y * w)) >= high) {
        *(gradient_out + x + (y * w)) = 255.f;
    } else if (*(gradient_in + x + (y * w)) < high
        && *(gradient_in + x + (y * w)) >= low) {
        *(gradient_out + x + (y * w)) = 125.f;
    } else {
        *(gradient_out + x + (y * w)) = 0.f;
    }
}

__global__ void Hysteresis(float* src,
                           float* dest,
                           int w,
                           int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    bool strong = false;

    *(dest + x + (y * w)) = *(src + x + (y * w));
    if (*(src + x + (y * w)) != 125.f) { return; }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int ix = x + i;
            int jy = y + j;
            if (ix < 0) { ix = 0; }
            if (ix >= w) { ix = w - 1; }
            if (jy < 0) { jy = 0; }
            if (jy >= h) { jy = h - 1; }

            if (*(src + ix + (jy * w)) == 255.f) { strong = true; }
        }
    }
    if (strong) {
        *(dest + x + (y * w)) = 255.f;
    } else {
        *(dest + x + (y * w)) = 0;
    }
}

std::shared_ptr<uint8_t> CudaCannyDetector::Detect() {
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

    float* kernel;
    float* tangent;

    dim3 threads(32, 32);
    dim3 block
        (m_w / threads.x + (m_w % threads.x == 0 ? 0 : 1),
         m_h / threads.y
             + (m_h % threads.y == 0 ? 0 : 1));

    cudaMalloc((void**) &kernel,
               sizeof(float) * m_gaussKernelSize * m_gaussKernelSize);
    cudaMalloc((void**) &dest1, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &dest2, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &tangent, sizeof(float) * m_w * m_h);
    dim3 gauss(m_gaussKernelSize, m_gaussKernelSize);
    cudaEventRecord(m_timers.All_start);

    cudaEventRecord(m_timers.GrayScale_start);
    convertToGreyScale<<<block, threads>>>(d_pixel, dest1, m_w, m_h);
    cudaEventRecord(m_timers.GrayScale_stop);
    cudaEventSynchronize(m_timers.GrayScale_stop);

    cudaEventRecord(m_timers.GaussCreation_start);
    GetGaussian<<<1, gauss>>>(kernel, m_gaussKernelSize, m_standardDeviation);
    cudaEventRecord(m_timers.GaussCreation_stop);
    cudaEventSynchronize(m_timers.GaussCreation_stop);

    cudaEventRecord(m_timers.Blur_start);
    GaussianFilter<<<block, threads>>>(dest1,
                                       dest2,
                                       kernel,
                                       m_gaussKernelSize,
                                       m_w,
                                       m_h);
    cudaEventRecord(m_timers.Blur_stop);
    cudaEventSynchronize(m_timers.Blur_stop);

    cudaEventRecord(m_timers.SobelOperator_start);
    DetectionOperator<<<block, threads>>>(dest2, dest1, tangent, m_w, m_h);
    cudaEventRecord(m_timers.SobelOperator_stop);
    cudaEventSynchronize(m_timers.SobelOperator_stop);

    cudaEventRecord(m_timers.NonMaximumSuppression_start);
    NonMaximumSuppression<<<block, threads>>>(dest1, dest2, tangent, m_w, m_h);
    cudaEventRecord(m_timers.NonMaximumSuppression_stop);
    cudaEventSynchronize(m_timers.NonMaximumSuppression_stop);

    cudaEventRecord(m_timers.DoubleThreshold_start);
    DoubleThreshold<<<block, threads>>>(dest2, dest1, m_w, m_h, m_high, m_low);
    cudaEventRecord(m_timers.DoubleThreshold_stop);
    cudaEventSynchronize(m_timers.DoubleThreshold_stop);

    cudaEventRecord(m_timers.Hysteresis_start);
    Hysteresis<<<block, threads>>>(dest1, dest2, m_w, m_h);
    cudaEventRecord(m_timers.Hysteresis_stop);
    cudaEventSynchronize(m_timers.Hysteresis_stop);

    CopyBack<<<block, threads>>>(d_pixel, dest2, m_w, m_h);
    cudaEventRecord(m_timers.All_stop);

    cudaEventSynchronize(m_timers.All_stop);

    cudaEventElapsedTime(&m_timings.All_ms,
                         m_timers.All_start,
                         m_timers.All_stop);

    cudaEventElapsedTime(&m_timings.GrayScale_ms,
                         m_timers.GrayScale_start,
                         m_timers.GrayScale_stop);

    cudaEventElapsedTime(&m_timings.GaussCreation_ms,
                         m_timers.GaussCreation_start,
                         m_timers.GaussCreation_stop);

    cudaEventElapsedTime(&m_timings.Blur_ms,
                         m_timers.Blur_start,
                         m_timers.Blur_stop);

    cudaEventElapsedTime(&m_timings.SobelOperator_ms,
                         m_timers.SobelOperator_start,
                         m_timers.SobelOperator_stop);

    cudaEventElapsedTime(&m_timings.NonMaximumSuppression_ms,
                         m_timers.NonMaximumSuppression_start,
                         m_timers.NonMaximumSuppression_stop);

    cudaEventElapsedTime(&m_timings.DoubleThreshold_ms,
                         m_timers.DoubleThreshold_start,
                         m_timers.DoubleThreshold_stop);

    cudaEventElapsedTime(&m_timings.Hysteresis_ms,
                         m_timers.Hysteresis_start,
                         m_timers.Hysteresis_stop);

    cudaMemcpy(m_detected, d_pixel,
               sizeof(uint8_t) * m_w * m_h * m_stride,
               cudaMemcpyDeviceToHost);

    cudaFree(dest1);
    cudaFree(dest2);
    cudaFree(kernel);
    cudaFree(tangent);
    cudaFree(d_pixel);
    cudaDeviceSynchronize();
    return std::shared_ptr<uint8_t>(m_detected);
}
