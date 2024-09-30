//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_

/*!
 * Simple struct containing timings for the Canny Edge detectors
 */
struct CannyTimings {
    float GrayScale_ms;
    float GaussCreation_ms;
    float Blur_ms;
    float SobelOperator_ms;
    float NonMaximumSuppression_ms;
    float DoubleThreshold_ms;
    float Hysteresis_ms;
    float All_ms;
};
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_CANNY_TIMINGS_H_
