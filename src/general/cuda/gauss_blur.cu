//
// Created by Palnit on 2024. 01. 21.
//

#include "general/cuda/gauss_blur.cuh"
#include <math_constants.h>
#include <cstdio>

__global__ void convertToGreyScale(uint8_t* base, float* dest, int w, int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    RGBA* color = (RGBA*) (base + (x * 4) + (y * w * 4));
    *(dest + x + (y * w)) = 0.299 * color->r
        + 0.587 * color->g
        + 0.114 * color->b;
}

__global__ void GetGaussian(float* kernel, int kernelSize, float sigma) {
    uint32_t x = threadIdx.x;
    uint32_t y = threadIdx.y;

    int k = (kernelSize - 1) / 2;

    float xp = (((x + 1.f) - (1.f + k)) * ((x + 1.f) - (1.f + k)));
    float yp = (((y + 1.f) - (1.f + k)) * ((y + 1.f) - (1.f + k)));
    *(kernel + x + (y * kernelSize)) =
        (1.f / (2.f * CUDART_PI_F * sigma * sigma))
            * exp(-((xp + yp) / (2.f * sigma * sigma)));
    __syncthreads();
    __shared__ float sum;
    if (x == 0 && y == 0) {
        sum = 0;

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                sum += *(kernel + i + (j * kernelSize));
            }
        }
    }
    __syncthreads();
    *(kernel + x + (y * kernelSize)) /= sum;

}

__global__ void GaussianFilter(float* img,
                               float* dest,
                               float* gauss,
                               int kernelSize,
                               int w,
                               int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }
    int k = (kernelSize - 1) / 2;

    float sum = 0;
    for (int i = -k; i <= k; i++) {
        for (int j = -k; j <= k; j++) {
            int ix = x + i;
            int jx = y + j;
            if (ix < 0) { ix = 0; }
            if (ix >= w) { ix = w - 1; }
            if (jx < 0) { jx = 0; }
            if (jx >= h) { jx = h - 1; }
            sum = std::fmaf(*(img + ix + (jx * w)),
                            *(gauss + (i + k)
                                + ((j + k) * kernelSize)),
                            sum);
        }
    }
    *(dest + x + (y * w)) = sum;
}

__global__ void CopyBack(uint8_t* src, float* dest, int w, int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }
    RGBA* color = (RGBA*) (src + (x * 4) + (y * w * 4));
    float value = roundf(*(dest + x + (y * w)));
    if (value < 0) {
        value = 0;
    }
    if (value > 255) {
        value = 255;
    }
    color->r = color->g = color->b = value;
}
