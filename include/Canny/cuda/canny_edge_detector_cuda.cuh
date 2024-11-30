//
// Created by Palnit on 2023. 11. 12.
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "Canny/canny_timings.h"
#include "Canny/canny_edge_detector.h"

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

class CannyEdgeDetectorCuda : public CannyEdgeDetector {
public:

    CannyEdgeDetectorCuda() = default;

    /*!
     * Implementation of the detect virtual function
     * \return the detected pixels
     */
    std::shared_ptr<uint8_t> Detect() override;

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
            cudaEventCreate(&GrayScale_start, cudaEventBlockingSync);
            cudaEventCreate(&GrayScale_stop, cudaEventBlockingSync);
            cudaEventCreate(&GaussCreation_start, cudaEventBlockingSync);
            cudaEventCreate(&GaussCreation_stop, cudaEventBlockingSync);
            cudaEventCreate(&Blur_start, cudaEventBlockingSync);
            cudaEventCreate(&Blur_stop, cudaEventBlockingSync);
            cudaEventCreate(&SobelOperator_start, cudaEventBlockingSync);
            cudaEventCreate(&SobelOperator_stop, cudaEventBlockingSync);
            cudaEventCreate(&NonMaximumSuppression_start,
                            cudaEventBlockingSync);
            cudaEventCreate(&NonMaximumSuppression_stop, cudaEventBlockingSync);
            cudaEventCreate(&DoubleThreshold_start, cudaEventBlockingSync);
            cudaEventCreate(&DoubleThreshold_stop, cudaEventBlockingSync);
            cudaEventCreate(&Hysteresis_start, cudaEventBlockingSync);
            cudaEventCreate(&Hysteresis_stop, cudaEventBlockingSync);
            cudaEventCreate(&All_start, cudaEventBlockingSync);
            cudaEventCreate(&All_stop, cudaEventBlockingSync);
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

    CudaTimers m_timers;
};
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
