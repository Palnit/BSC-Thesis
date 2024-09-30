//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "include/Dog/dog_timings.h"

/*!
 * This cuda function calculates the difference of 2 gaussian kernels at every
 * point of the matrix
 * \param kernel1 The first kernel
 * \param kernel2 The second kernel
 * \param finalKernel The output kernel
 * \param kernelSize The size of the kernels
 */
__global__ void DifferenceOfGaussian(float* kernel1,
                                     float* kernel2,
                                     float* finalKernel,
                                     int kernelSize);

/*!
 * \class CudaDogDetector
 * \brief A utility class to temporarily store the data related to the algorithm
 *
 * This class simply exist so that the memory allocation on the cuda side
 * is taken care of and so that a class can call the cuda algorithms easily and
 * this keeps the normal cpp and cuda separate
 */
class CudaDogDetector {
public:

    /*!
     * Constructor
     * \param src The source image
     * \param w The width of the image
     * \param h The height of the image
     * \param gaussKernelSize The size of the 2 gaussian kernels
     * \param standardDeviation1 The standard deviation of the 1. kernel
     * \param standardDeviation2 The standard deviation of the 2. kernel
     */
    CudaDogDetector(uint8_t* src,
                    int w,
                    int h,
                    int gaussKernelSize,
                    float standardDeviation1,
                    float standardDeviation2) :
        m_src(src),
        m_w(w),
        m_h(h),
        m_gaussKernelSize(gaussKernelSize),
        m_standardDeviation1(standardDeviation1),
        m_standardDeviation2(standardDeviation2) {
        DogDetect();
    }

    /*!
     * Function to get the timings of the algorithms
     * \return The timings that has been calculated
     */
    DogTimings GetTimings() {
        return m_timings;
    }
private:

    /*!
     * \class CudaTimers
     * \brief Utility class to keep the cuda events for timings separate
     *
     * It creates and destroys the cuda events that are needed to calculate the
     * running time of the algorithm
     */
    struct CudaTimers {

        /*!
         * Constructor creates the cuda events
         */
        CudaTimers() {
            cudaEventCreate(&All_start);
            cudaEventCreate(&All_stop);
            cudaEventCreate(&GrayScale_start);
            cudaEventCreate(&GrayScale_stop);
            cudaEventCreate(&Gauss1Creation_start);
            cudaEventCreate(&Gauss1Creation_stop);
            cudaEventCreate(&Gauss2Creation_start);
            cudaEventCreate(&Gauss2Creation_stop);
            cudaEventCreate(&DifferenceOfGaussian_start);
            cudaEventCreate(&DifferenceOfGaussian_stop);
            cudaEventCreate(&Convolution_start);
            cudaEventCreate(&Convolution_stop);
        }

        /*!
         * Destructor deletes the cuda events
         */
        ~CudaTimers() {
            cudaEventDestroy(All_start);
            cudaEventDestroy(All_stop);
            cudaEventDestroy(GrayScale_start);
            cudaEventDestroy(GrayScale_stop);
            cudaEventDestroy(Gauss1Creation_start);
            cudaEventDestroy(Gauss1Creation_stop);
            cudaEventDestroy(Gauss2Creation_start);
            cudaEventDestroy(Gauss2Creation_stop);
            cudaEventDestroy(DifferenceOfGaussian_start);
            cudaEventDestroy(DifferenceOfGaussian_stop);
            cudaEventDestroy(Convolution_start);
            cudaEventDestroy(Convolution_stop);
        }
        cudaEvent_t All_start;
        cudaEvent_t All_stop;
        cudaEvent_t GrayScale_start;
        cudaEvent_t GrayScale_stop;
        cudaEvent_t Gauss1Creation_start;
        cudaEvent_t Gauss1Creation_stop;
        cudaEvent_t Gauss2Creation_start;
        cudaEvent_t Gauss2Creation_stop;
        cudaEvent_t DifferenceOfGaussian_start;
        cudaEvent_t DifferenceOfGaussian_stop;
        cudaEvent_t Convolution_start;
        cudaEvent_t Convolution_stop;
    };

    /*!
     * The main function that class the cuda algorithms in order for the edge
     * detection to happen
     */
    void DogDetect();

    uint8_t* m_src;
    int m_w;
    int m_h;
    int m_gaussKernelSize;
    float m_standardDeviation1;
    float m_standardDeviation2;
    CudaTimers m_timers;
    DogTimings m_timings;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
