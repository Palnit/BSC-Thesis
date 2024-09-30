//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_

#include <cuda_runtime.h>
#include <cstdint>

/*!
 * Struct used to handel uint32_t types as RGBA when casted to this struct
 */
typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;


/*!
 * A cuda global function that for every thread takes the corresponding block
 * and threads pixel form the base array of pixel of a picture and converts it
 * to grey a grey scale image and writes its data to the dest float
 * \param base The base image
 * \param dest The output image
 * \param w The width of the image
 * \param h The height of the image
 */
__global__ void convertToGreyScale(uint8_t* base, float* dest, int w, int h);

/*!
 * A cuda global function that for every thread calculates the corresponding
 * element in the kernel matrix and calculates a Gaussian kernels data there
 * \param kernel The Gaussian kernel of output
 * \param kernelSize The size of the output
 * \param sigma The sigma of the kernel
 */
__global__ void GetGaussian(float* kernel, int kernelSize, float sigma);

/*!
 * A cuda global function that for every threads takes the corresponding block
 * and threads pixel from the src and calculates the convolution with tbe gauss
 * matrix and writes the output to the dest float array
 * \param src The base image
 * \param dest The output image
 * \param gauss The convolution kernel
 * \param kernelSize The size of the convolution kernel
 * \param w The width of the image
 * \param h  The height of the image
 */
__global__ void GaussianFilter(float* src,
                               float* dest,
                               float* gauss,
                               int kernelSize,
                               int w,
                               int h);

/*!
 * A cuda global function that for every threads takes the corresponding block
 * and threads pixel from the src and copy it to the uint8_t dest array
 * \param dest The destination image
 * \param src The src image
 * \param w The width of the image
 * \param h The height of the image
 */
__global__ void CopyBack(uint8_t* dest, float* src, int w, int h);

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_
