//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_DOG_DOG_TIMINGS_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_DOG_DOG_TIMINGS_H_

#include "general/timings_base.h"
/*!
 * Simple struct containing timings for the Different of Gaussian Edge detectors
 */
struct DogTimings : TimingsBase {
    float GrayScale_ms;
    float Gauss1Creation_ms;
    float Gauss2Creation_ms;
    float DifferenceOfGaussian_ms;
    float Convolution_ms;
};
#endif//GPGPU_EDGE_DETECTOR_INCLUDE_DOG_DOG_TIMINGS_H_
