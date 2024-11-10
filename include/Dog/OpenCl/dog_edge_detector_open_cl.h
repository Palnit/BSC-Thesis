#ifndef BSC_THESIS_INCLUDE_DOG_OPENCL_DOG_EDGE_DETECTOR_OPEN_CL_H_
#define BSC_THESIS_INCLUDE_DOG_OPENCL_DOG_EDGE_DETECTOR_OPEN_CL_H_

#include "general/detector_base.h"
#include "Dog/dog_timings.h"

class DogEdgeDetectorOpenCl : public DetectorBase {
public:
    DogEdgeDetectorOpenCl(SDL_Surface* base, std::string name)
        : DetectorBase(base, std::move(name)) {}

    void DetectEdge() override;

    void DisplayImGui() override;

    void Display() override;

private:
    int m_gaussKernelSize = 17;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 10;
    bool m_timingsReady = false;
    DogTimings m_timings;
};

#endif //BSC_THESIS_INCLUDE_DOG_OPENCL_DOG_EDGE_DETECTOR_OPEN_CL_H_
