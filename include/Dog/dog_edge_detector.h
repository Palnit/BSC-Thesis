#ifndef BSC_THESIS_DOG_EDGE_DETECTOR_H
#define BSC_THESIS_DOG_EDGE_DETECTOR_H

#include "dog_timings.h"
#include "general/edge_detector_base.h"

class DogEdgeDetector : public EdgeDetectorBase<DogTimings> {
public:
    DogEdgeDetector() {}

    DogEdgeDetector(uint8_t* pixel,
                    int w,
                    int h,
                    int gaussKernelSize,
                    float standardDeviation1,
                    float standardDeviation2)
        : EdgeDetectorBase<DogTimings>(pixel, w, h),
          m_gaussKernelSize(gaussKernelSize),
          m_standardDeviation1(standardDeviation1),
          m_standardDeviation2(standardDeviation2) {}

    int* getGaussKernelSize() { return &m_gaussKernelSize; }
    float* getStandardDeviation1() { return &m_standardDeviation1; }
    float* getStandardDeviation2() { return &m_standardDeviation2; }

protected:
    int m_gaussKernelSize = 7;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 10;
};

#endif//BSC_THESIS_DOG_EDGE_DETECTOR_H
