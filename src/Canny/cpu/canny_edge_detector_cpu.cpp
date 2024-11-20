#include "Canny/cpu/canny_edge_detector_cpu.h"
#include <chrono>
#include <functional>
#include <memory>
#include "general/cpu/gauss_blur_cpu.h"
#define M_PI_F 3.141592654F

void DetectorsCPU::DetectionOperator(float* src,
                                     float* dest,
                                     float* tangent,
                                     int w,
                                     int h) {
    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
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
            float angle = (atan2(SumX, SumY) * 180.f) / M_PI_F;
            if (angle < 0) { angle += 180; }
            *(tangent + x + (y * w)) = angle;
        }
    }
}
void DetectorsCPU::NonMaximumSuppression(float* src,
                                         float* dest,
                                         float* tangent,
                                         int w,
                                         int h) {
    float* tangentA;
    float gradientA;
    float gradientP;
    float gradientN;
    int yp;
    int yn;
    int xp;
    int xn;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {

            tangentA = (tangent + x + (y * w));
            gradientA = *(src + x + (y * w));
            gradientP = 2000;
            gradientN = 2000;

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
    }
}
void DetectorsCPU::DoubleThreshold(float* src,
                                   float* dest,
                                   int w,
                                   int h,
                                   float high,
                                   float low) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            if (*(src + x + (y * w)) >= high) {
                *(dest + x + (y * w)) = 255.f;
            } else if (*(src + x + (y * w)) < high
                       && *(src + x + (y * w)) >= low) {
                *(dest + x + (y * w)) = 125.f;
            } else {
                *(dest + x + (y * w)) = 0.f;
            }
        }
    }
}
void DetectorsCPU::Hysteresis(float* src, float* dest, int w, int h) {
    int k = 1;
    bool strong = false;
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            *(dest + x + (y * w)) = *(src + x + (y * w));
            if (*(src + x + (y * w)) != 125.f) { continue; }
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
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
    }
}

std::shared_ptr<uint8_t> CannyEdgeDetectorCPU::Detect() {
    m_detected =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) * m_w * m_h * m_stride));
    float* m_pixels1 = static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    float* m_pixels2 = static_cast<float*>(malloc(sizeof(float) * m_w * m_h));
    float* m_kernel = static_cast<float*>(
        malloc(sizeof(float) * m_gaussKernelSize * m_gaussKernelSize));
    float* m_tangent = static_cast<float*>(malloc(sizeof(float) * m_w * m_h));

    auto t1 = std::chrono::high_resolution_clock::now();
    m_timings.GrayScale_ms = Detectors::TimerRunner(
        DetectorsCPU::ConvertGrayScale, m_pixels, m_pixels1, m_w, m_h);
    m_timings.GaussCreation_ms =
        Detectors::TimerRunner(DetectorsCPU::GenerateGauss, m_kernel,
                               m_gaussKernelSize, m_standardDeviation);

    m_timings.Blur_ms = Detectors::TimerRunner(DetectorsCPU::GaussianFilter,
                                               m_pixels1, m_pixels2, m_kernel,
                                               m_gaussKernelSize, m_w, m_h);
    m_timings.SobelOperator_ms =
        Detectors::TimerRunner(DetectorsCPU::DetectionOperator, m_pixels2,
                               m_pixels1, m_tangent, m_w, m_h);
    m_timings.NonMaximumSuppression_ms =
        Detectors::TimerRunner(DetectorsCPU::NonMaximumSuppression, m_pixels1,
                               m_pixels2, m_tangent, m_w, m_h);
    m_timings.DoubleThreshold_ms =
        Detectors::TimerRunner(DetectorsCPU::DoubleThreshold, m_pixels2,
                               m_pixels1, m_w, m_h, m_high, m_low);
    m_timings.Hysteresis_ms = Detectors::TimerRunner(
        DetectorsCPU::Hysteresis, m_pixels1, m_pixels2, m_w, m_h);

    DetectorsCPU::CopyBack(m_detected, m_pixels2, m_w, m_h);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> time = t2 - t1;
    m_timings.All_ms = time.count();
    free(m_pixels1);
    free(m_pixels2);
    free(m_kernel);
    free(m_tangent);
    return std::shared_ptr<uint8_t>(m_detected);
}
