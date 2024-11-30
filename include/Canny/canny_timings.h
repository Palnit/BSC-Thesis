#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_

#include "general/timings_base.h"
/*!
 * Simple struct containing timings for the Canny Edge detectors
 */
struct CannyTimings : TimingsBase {
    float GrayScale_ms;
    float GaussCreation_ms;
    float Blur_ms;
    float SobelOperator_ms;
    float NonMaximumSuppression_ms;
    float DoubleThreshold_ms;
    float Hysteresis_ms;
};
#endif//GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_
