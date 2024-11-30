//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "Dog/dog_timings.h"
#include "Dog/dog_edge_detector.h"

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

class DogEdgeDetectorCuda : public DogEdgeDetector {
public:

    DogEdgeDetectorCuda() = default;


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
            cudaEventCreate(&All_start, cudaEventBlockingSync);
            cudaEventCreate(&All_stop, cudaEventBlockingSync);
            cudaEventCreate(&GrayScale_start, cudaEventBlockingSync);
            cudaEventCreate(&GrayScale_stop, cudaEventBlockingSync);
            cudaEventCreate(&Gauss1Creation_start, cudaEventBlockingSync);
            cudaEventCreate(&Gauss1Creation_stop, cudaEventBlockingSync);
            cudaEventCreate(&Gauss2Creation_start, cudaEventBlockingSync);
            cudaEventCreate(&Gauss2Creation_stop, cudaEventBlockingSync);
            cudaEventCreate(&DifferenceOfGaussian_start, cudaEventBlockingSync);
            cudaEventCreate(&DifferenceOfGaussian_stop, cudaEventBlockingSync);
            cudaEventCreate(&Convolution_start, cudaEventBlockingSync);
            cudaEventCreate(&Convolution_stop, cudaEventBlockingSync);
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

    CudaTimers m_timers;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
