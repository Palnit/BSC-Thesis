//
// Created by Palnit on 2024. 01. 22.
//
#include "general/cpu/gauss_blur_cpu.h"
#include <cmath>
#include <iostream>
#include "general/OpenGL_SDL/generic_structs.h"
#define M_PI_F 3.141592654F

void DetectorsCPU::CopyBack(uint8_t* dest, float* src, int w, int h) {
    RGBA* color;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            color = (RGBA*) (dest + (x * 4) + (y * w * 4));
            float value = roundf(*(src + x + (y * w)));
            if (value < 0) {
                value = 0;
            }
            if (value > 255) {
                value = 255;
            }
            color->r = color->g = color->b = value;
        }
    }
}

void DetectorsCPU::ConvertGrayScale(uint8_t* base, float* dest, int w, int h) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            RGBA* color = (RGBA*) (base + (x * 4) + (y * w * 4));
            *(dest + x + (y * w)) =
                0.299 * color->r + 0.587 * color->g + 0.114 * color->b;
        }
    }
}

void DetectorsCPU::GenerateGauss(float* kernel, int kernelSize, float sigma) {
    int k = (kernelSize - 1) / 2;
    float sum = 0;
    for (int x = 0; x < kernelSize; ++x) {
        for (int y = 0; y < kernelSize; ++y) {
            float xp = (((x + 1.f) - (1.f + k)) * ((x + 1.f) - (1.f + k)));
            float yp = (((y + 1.f) - (1.f + k)) * ((y + 1.f) - (1.f + k)));
            *(kernel + x + (y * kernelSize)) =
                (1.f / (2.f * M_PI_F * sigma * sigma))
                    * exp(-((xp + yp) / (2.f * sigma * sigma)));
            sum += *(kernel + x + (y * kernelSize));
        }
    }
    for (int x = 0; x < kernelSize; ++x) {
        for (int y = 0; y < kernelSize; ++y) {
            *(kernel + x + (y * kernelSize)) /= sum;
        }
    }
}
void DetectorsCPU::GaussianFilter(float* img,
                                  float* dest,
                                  float* gauss,
                                  int kernelSize,
                                  int w,
                                  int h) {

    int k = (kernelSize - 1) / 2;

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
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
                                    *(gauss + (i + k) + ((j + k) * kernelSize)),
                                    sum);
                }
            }
            *(dest + x + (y * w)) = sum;
        }
    }
}
