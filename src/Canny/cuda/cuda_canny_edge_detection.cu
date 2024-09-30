//
// Created by Palnit on 2023. 11. 12.
//

#include "include/Canny/cuda/cuda_canny_edge_detection.cuh"
#include <cstdio>
#include <math_constants.h>
#include "include/general/cuda/gauss_blur.cuh"

__global__ void DetectionOperator(float* src,
                                  float* gradient,
                                  float* tangent,
                                  int w,
                                  int h) {
    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] = *(src + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};

    float SumX = 0;
    float SumY = 0;

    if (threadIdx.x > 1 - 1 && threadIdx.y > 1 - 1 && threadIdx.x < 32 - 1
        && threadIdx.y < 32 - 1 && col_i < w && row_i < h) {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                SumX = fmaf(src_shared[threadIdx.x + i][threadIdx.y + j],
                            (*(SobelX + (i + 1) + ((j + 1) * 3))),
                            SumX);
                SumY = fmaf(src_shared[threadIdx.x + i][threadIdx.y + j],
                            (*(SobelY + (i + 1) + ((j + 1) * 3))),
                            SumY);
            }
        }
        *(gradient + col_i + (row_i * w)) = hypotf(SumX, SumY);
        float angle = (atan2(SumX, SumY) * 180.f) / CUDART_PI_F;
        if (angle < 0) {
            angle += 180;
        }
        *(tangent + col_i + (row_i * w)) = angle;
    }

}

__global__ void NonMaximumSuppression(float* gradient_in,
                                      float* gradient_out,
                                      float* tangent,
                                      int w,
                                      int h) {

    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] =
            *(gradient_in + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 2000;
    }

    __syncthreads();

    if (threadIdx.x <= 1 - 1 || threadIdx.y <= 1 - 1 || threadIdx.x >= 32 - 1
        || threadIdx.y >= 32 - 1 || col_i >= w || row_i >= h) {
        return;
    }

    float* tangentA = (tangent + col_i + (row_i * w));
    float gradientA = src_shared[threadIdx.x][threadIdx.y];
    float gradientP = 2000;
    float gradientN = 2000;

    if ((0 <= *tangentA && *tangentA < 22.5)
        || (157.5 <= *tangentA && *tangentA <= 180)) {
        gradientP = src_shared[threadIdx.x][threadIdx.y + 1];
        gradientN = src_shared[threadIdx.x][threadIdx.y - 1];
    } else if (22.5 <= *tangentA && *tangentA < 67.5) {
        gradientP = src_shared[threadIdx.x + 1][threadIdx.y - 1];
        gradientN = src_shared[threadIdx.x - 1][threadIdx.y + 1];
    } else if (67.5 <= *tangentA && *tangentA < 112.5) {
        gradientP = src_shared[threadIdx.x + 1][threadIdx.y];
        gradientN = src_shared[threadIdx.x - 1][threadIdx.y];
    } else if (112.5 <= *tangentA && *tangentA < 157.5) {
        gradientP = src_shared[threadIdx.x - 1][threadIdx.y - 1];
        gradientN = src_shared[threadIdx.x + 1][threadIdx.y + 1];
    }

    if (gradientA < gradientN || gradientA < gradientP) {
        gradientA = 0.f;
    }

    *(gradient_out + col_i + (row_i * w)) = gradientA;
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

__global__ void Hysteresis(float* gradient_in,
                           float* gradient_out,
                           int w,
                           int h) {
    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] =
            *(gradient_in + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    if (threadIdx.x <= 1 - 1 || threadIdx.y <= 1 - 1 || threadIdx.x >= 32 - 1
        || threadIdx.y >= 32 - 1 || col_i >= w || row_i >= h) {
        return;
    }

    bool strong = false;

    float gradientA = src_shared[threadIdx.x][threadIdx.y];

    *(gradient_out + col_i + (row_i * w)) = gradientA;
    if (gradientA != 125.f) {
        return;
    }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            if (src_shared[threadIdx.x + j][threadIdx.y + j] == 255.f) {
                strong = true;
            }
        }
    }
    if (strong) {
        gradientA = 255.f;
    } else {
        gradientA = 0.f;
    }

    *(gradient_out + col_i + (row_i * w)) = gradientA;
}

void CudaCannyDetector::CannyEdgeDetection() {
    float* dest1;
    float* dest2;

    float* kernel;
    float* tangent;

    dim3 threads(32, 32);
    dim3 block
        (m_w / threads.x + (m_w % threads.x == 0 ? 0 : 1),
         m_h / threads.y
             + (m_h % threads.y == 0 ? 0 : 1));

    dim3 block2
        ((m_w / (threads.x - m_gaussKernelSize + 1))
             + (m_w % (threads.x - m_gaussKernelSize + 1) == 0 ? 0 : 1),
         (m_h / (threads.y - m_gaussKernelSize + 1))
             + (m_h % (threads.y - m_gaussKernelSize + 1) == 0 ? 0 : 1));
    dim3 block3
        ((m_w / (threads.x - 3 + 1)) + (m_w % (threads.x - 3 + 1) == 0 ? 0 : 1),
         (m_h / (threads.y - 3 + 1))
             + (m_h % (threads.y - 3 + 1) == 0 ? 0 : 1));

    cudaMalloc((void**) &kernel,
               sizeof(float) * m_gaussKernelSize * m_gaussKernelSize);
    cudaMalloc((void**) &dest1, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &dest2, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &tangent, sizeof(float) * m_w * m_h);
    dim3 gauss(m_gaussKernelSize, m_gaussKernelSize);
    cudaEventRecord(m_timers.All_start);

    cudaEventRecord(m_timers.GrayScale_start);
    convertToGreyScale<<<block, threads>>>(m_src, dest1, m_w, m_h);
    cudaEventRecord(m_timers.GrayScale_stop);
    cudaEventSynchronize(m_timers.GrayScale_stop);

    cudaEventRecord(m_timers.GaussCreation_start);
    GetGaussian<<<1, gauss>>>(kernel, m_gaussKernelSize, m_standardDeviation);
    cudaEventRecord(m_timers.GaussCreation_stop);
    cudaEventSynchronize(m_timers.GaussCreation_stop);

    cudaEventRecord(m_timers.Blur_start);
    GaussianFilter<<<block2, threads>>>(dest1,
                                        dest2,
                                        kernel,
                                        m_gaussKernelSize,
                                        m_w,
                                        m_h);
    cudaEventRecord(m_timers.Blur_stop);
    cudaEventSynchronize(m_timers.Blur_stop);

    cudaEventRecord(m_timers.SobelOperator_start);
    DetectionOperator<<<block3, threads>>>(dest2, dest1, tangent, m_w, m_h);
    cudaEventRecord(m_timers.SobelOperator_stop);
    cudaEventSynchronize(m_timers.SobelOperator_stop);

    cudaEventRecord(m_timers.NonMaximumSuppression_start);
    NonMaximumSuppression<<<block3, threads>>>(dest1, dest2, tangent, m_w, m_h);
    cudaEventRecord(m_timers.NonMaximumSuppression_stop);
    cudaEventSynchronize(m_timers.NonMaximumSuppression_stop);

    cudaEventRecord(m_timers.DoubleThreshold_start);
    DoubleThreshold<<<block, threads>>>(dest2, dest1, m_w, m_h, m_high, m_low);
    cudaEventRecord(m_timers.DoubleThreshold_stop);
    cudaEventSynchronize(m_timers.DoubleThreshold_stop);

    cudaEventRecord(m_timers.Hysteresis_start);
    Hysteresis<<<block3, threads>>>(dest1, dest2, m_w, m_h);
    cudaEventRecord(m_timers.Hysteresis_stop);
    cudaEventSynchronize(m_timers.Hysteresis_stop);

    CopyBack<<<block, threads>>>(m_src, dest2, m_w, m_h);
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

    cudaFree(dest1);
    cudaFree(dest2);
    cudaFree(kernel);
    cudaFree(tangent);
    cudaDeviceSynchronize();
}
