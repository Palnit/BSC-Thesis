//
// Created by Palnit on 2024. 01. 22.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_

#include <cstdint>
#include <functional>
#include <chrono>

/*!
 * A namespace containing the implementation of the cpu implementation of the
 * edge detection algorithm
 */
namespace DetectorsCPU {

/*!
 * A function to copy the float data back to the uint8_t data of an SDL_Surface
 * \param dest The destination surface pixels
 * \param src The source data of the pixels
 * \param w The width of the image
 * \param h The height of the image
 */
void CopyBack(uint8_t* dest, float* src, int w, int h);

/*!
 * Converts an image to gray scale
 * \param base The base image to be converted
 * \param dest The output image
 * \param w The width of the image
 * \param h The height of the image
 */
void ConvertGrayScale(uint8_t* base, float* dest, int w, int h);

/*!
 * Generates a gaussian kernel
 * \param kernel The generated kernel
 * \param kernelSize The size of the kernel
 * \param sigma The standard deviation of the kernel
 */
void GenerateGauss(float* kernel, int kernelSize, float sigma);

/*!
 * A gaussian filter convolution of an image
 * \param img The image to be convoluted
 * \param dest The destination image
 * \param gauss The gaussian kernel
 * \param kernelSize The size of the kernel
 * \param w The width of the kernel
 * \param h The height of the kernel
 */
void GaussianFilter(float* img,
                    float* dest,
                    float* gauss,
                    int kernelSize,
                    int w,
                    int h);

/*!
 * A function that takes any kinds of function and it's arguments as an argument
 * runs it and measures the running time of the argument function with
 * std::chrono::high_resolution_clock and returns the time
 * \tparam F The functions type
 * \tparam Args The argument list
 * \param func The function to be measured
 * \param args The parameters of the function
 * \return
 */
template<typename F, typename ... Args>
float TimerRunner(F&& func, Args&& ... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> chrono = t2 - t1;
    return chrono.count();
}
}
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_
