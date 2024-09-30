//
// Created by Palnit on 2024. 01. 22.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_
#define GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_

#include "include/general/detector_base.h"
#include "include/Dog/dog_timings.h"

/*!
 * \class DogEdgeDetectorCPU
 * \brief Implementation of the DetectorBase class for Dog detection on cpu
 *
 * It implements the base class and stores data related to the dog edge detection
 */
class DogEdgeDetectorCPU : public DetectorBase {
public:

    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     */
    DogEdgeDetectorCPU(SDL_Surface* base, std::string name) : DetectorBase(base,
                                                                           std::move(
                                                                               name)),
                                                              m_w(m_base->w),
                                                              m_h(m_base->h) {}
    /*!
     * Implementation of the DetectEdge function class the detection functions
     */
    void DetectEdge() override;

    /*!
     * Implementation of the DisplayImGui function displays the variables
     * related to this edge detection method to be modified easily
     */
    void DisplayImGui() override;

    /*!
     * Implementation of the Display function displays the base and
     * detected image
     */
    void Display() override;
private:
    int m_w;
    int m_h;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel1;
    float* m_kernel2;
    float* m_finalKernel;
    int m_gaussKernelSize = 7;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 0.7;
    bool m_timingsReady = false;
    DogTimings m_timings;
};

/*!
 * A namespace containing the implementation of the cpu implementation of the
 * edge detection algorithm
 */
namespace DetectorsCPU {
/*!
 * This function calculates the difference of 2 gaussian kernels
 * \param kernel1 The first kernel
 * \param kernel2 The second kernel
 * \param finalKernel The output kernel
 * \param kernelSize The size of the kernels
 */
void DifferenceOfGaussian(float* kernel1,
                          float* kernel2,
                          float* finalKernel,
                          int kernelSize);
}

#endif //GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_
