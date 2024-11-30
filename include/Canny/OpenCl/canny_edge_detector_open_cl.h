#ifndef BSC_THESIS_CANNY_EDGE_DETECTOR_OPEN_CL_H
#define BSC_THESIS_CANNY_EDGE_DETECTOR_OPEN_CL_H

#include "Canny/canny_edge_detector.h"

class CannyEdgeDetectorOpenCl : public CannyEdgeDetector {
public:
    CannyEdgeDetectorOpenCl() = default;

    /*!
     * Implementation of the detect virtual function
     * \return the detected pixels
     */
    std::shared_ptr<uint8_t> Detect() override;
};

#endif//BSC_THESIS_CANNY_EDGE_DETECTOR_OPEN_CL_H
