#ifndef BSC_THESIS_DOG_EDGE_DETECTOR_OPEN_CL_H
#define BSC_THESIS_DOG_EDGE_DETECTOR_OPEN_CL_H

#include "Dog/dog_edge_detector.h"
class DogEdgeDetectorOpenCl : public DogEdgeDetector {
public:
    DogEdgeDetectorOpenCl() = default;

    /*!
     * Implementation of the detect virtual function
     * \return the detected pixels
     */
    std::shared_ptr<uint8_t> Detect() override;
};

#endif//BSC_THESIS_DOG_EDGE_DETECTOR_OPEN_CL_H
