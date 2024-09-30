//
// Created by Palnit on 2023. 11. 12.
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "include/Canny/canny_timings.h"

/*!
 * This function uses the sobel operator to calculate the gradient and the
 * tangent for every pixel of the image
 * \param src The source grey scaled image
 * \param gradient The output gradient
 * \param tangent The output tangent
 * \param w The width of the image
 * \param h The height of the image
 */
__global__ void DetectionOperator(float* src,
                                  float* gradient,
                                  float* tangent,
                                  int w,
                                  int h);

/*!
 * This function keeps the current pixel value if it's the maximum gradient in
 * the tangent direction
 * \param gradient_in The input gradient
 * \param gradient_out The output gradient
 * \param tangent The input's tangent
 * \param w The width of the image
 * \param h The height of the image
 */
__global__ void NonMaximumSuppression(float* gradient_in,
                                      float* gradient_out,
                                      float* tangent,
                                      int w,
                                      int h);

/*!
 * This function defines strong and week edges based on 2 arbitrary thresholds
 * \param gradient_in The input gradient
 * \param gradient_out The output gradient
 * \param w The width of the image
 * \param h The height of the image
 * \param high The high threshold
 * \param low The low threshold
 */
__global__ void DoubleThreshold(float* gradient_in,
                                float* gradient_out,
                                int w,
                                int h,
                                float high,
                                float low);

/*!
 * This function keeps the week edges if they have at least one strong edge
 * adjacent to them
 * \param gradient_in The input gradient
 * \param gradient_out The output gradient
 * \param high The high threshold
 * \param low The low threshold
 */
__global__ void Hysteresis(float* gradient_in,
                           float* gradient_out,
                           int w,
                           int h);

/*!
 * \class CudaDogDetector
 * \brief A utility class to temporarily store the data related to the algorithm
 *
 * This class simply exist so that the memory allocation on the cuda side
 * is taken care of and so that a class can call the cuda algorithms easily and
 * this keeps the normal cpp and cuda separate
 */
class CudaCannyDetector {
public:

    /*!
     * Constructor
     * \param src The source image
     * \param w The width of the image
     * \param h The height of the image
     * \param gaussKernelSize The size of the gaussian filter
     * \param standardDeviation The standard deviation of the gaussian filter
     * \param high The high threshold for double thresholding
     * \param low The low threshold for double thresholding
     */
    CudaCannyDetector(uint8_t* src,
                      int w,
                      int h,
                      int gaussKernelSize,
                      float standardDeviation,
                      float high,
                      float low)
        : m_src(src),
          m_w(w),
          m_h(h),
          m_gaussKernelSize(gaussKernelSize),
          m_standardDeviation(standardDeviation),
          m_high(high),
          m_low(low) {
        CannyEdgeDetection();
    }

    /*!
     * Function to get the timings of the algorithms
     * \return The timings that has been calculated
     */
    CannyTimings GetTimings() {
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
            cudaEventCreate(&GrayScale_start);
            cudaEventCreate(&GrayScale_stop);
            cudaEventCreate(&GaussCreation_start);
            cudaEventCreate(&GaussCreation_stop);
            cudaEventCreate(&Blur_start);
            cudaEventCreate(&Blur_stop);
            cudaEventCreate(&SobelOperator_start);
            cudaEventCreate(&SobelOperator_stop);
            cudaEventCreate(&NonMaximumSuppression_start);
            cudaEventCreate(&NonMaximumSuppression_stop);
            cudaEventCreate(&DoubleThreshold_start);
            cudaEventCreate(&DoubleThreshold_stop);
            cudaEventCreate(&Hysteresis_start);
            cudaEventCreate(&Hysteresis_stop);
            cudaEventCreate(&All_start);
            cudaEventCreate(&All_stop);
        }

        /*!
         * Destructor deletes the cuda events
         */
        ~CudaTimers() {
            cudaEventDestroy(GrayScale_start);
            cudaEventDestroy(GrayScale_stop);
            cudaEventDestroy(GaussCreation_start);
            cudaEventDestroy(GaussCreation_stop);
            cudaEventDestroy(Blur_start);
            cudaEventDestroy(Blur_stop);
            cudaEventDestroy(SobelOperator_start);
            cudaEventDestroy(SobelOperator_stop);
            cudaEventDestroy(NonMaximumSuppression_start);
            cudaEventDestroy(NonMaximumSuppression_stop);
            cudaEventDestroy(DoubleThreshold_start);
            cudaEventDestroy(DoubleThreshold_stop);
            cudaEventDestroy(Hysteresis_start);
            cudaEventDestroy(Hysteresis_stop);
            cudaEventDestroy(All_start);
            cudaEventDestroy(All_stop);
        }
        cudaEvent_t GrayScale_start;
        cudaEvent_t GrayScale_stop;
        cudaEvent_t GaussCreation_start;
        cudaEvent_t GaussCreation_stop;
        cudaEvent_t Blur_start;
        cudaEvent_t Blur_stop;
        cudaEvent_t SobelOperator_start;
        cudaEvent_t SobelOperator_stop;
        cudaEvent_t NonMaximumSuppression_start;
        cudaEvent_t NonMaximumSuppression_stop;
        cudaEvent_t DoubleThreshold_start;
        cudaEvent_t DoubleThreshold_stop;
        cudaEvent_t Hysteresis_start;
        cudaEvent_t Hysteresis_stop;
        cudaEvent_t All_start;
        cudaEvent_t All_stop;
    };

    /*!
     * The main function that class the cuda algorithms in order for the edge
     * detection to happen
     */
    void CannyEdgeDetection();

    uint8_t* m_src;
    int m_w;
    int m_h;
    int m_gaussKernelSize;
    float m_standardDeviation;
    float m_high;
    float m_low;
    CudaTimers m_timers;
    CannyTimings m_timings;
};
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
