#ifndef BSC_THESIS_DOG_EDGE_DETECTOR_CPU_H
#define BSC_THESIS_DOG_EDGE_DETECTOR_CPU_H

#include "Dog/dog_edge_detector.h"
class DogEdgeDetectorCPU : public DogEdgeDetector {
public:
    DogEdgeDetectorCPU() = default;
    std::shared_ptr<uint8_t> Detect() override;
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
}// namespace DetectorsCPU

#endif//BSC_THESIS_DOG_EDGE_DETECTOR_CPU_H
