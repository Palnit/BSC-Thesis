//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_

#include "include/general/detector_base.h"
#include "include/Dog/dog_timings.h"

/*!
 * \class DogEdgeDetectorCuda
 * \brief Implementation of the DetectorBase class for Dog detection on cuda
 *
 * It implements the base class and stores data related to the dog edge detection
 * and class the cuda functions
 */
class DogEdgeDetectorCuda : public DetectorBase {
public:

    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     */
    DogEdgeDetectorCuda(SDL_Surface* base,
                        std::string name) : DetectorBase(
        base,
        std::move(name)) {
    }

    /*!
     * Implementation of the DetectEdge function calls from the cuda library
     * and hides all cuda related things
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
    int m_gaussKernelSize = 17;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 10;
    bool m_timingsReady = false;
    DogTimings m_timings;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_
