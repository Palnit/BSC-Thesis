#ifndef BSC_THESIS_CANNY_EDGE_DETECTOR_H
#define BSC_THESIS_CANNY_EDGE_DETECTOR_H

#include "canny_timings.h"
#include "general/edge_detector_base.h"
class CannyEdgeDetector : public EdgeDetectorBase<CannyTimings> {
public:
    CannyEdgeDetector() {}
    CannyEdgeDetector(uint8_t* pixel,
                      int w,
                      int h,
                      int gaussKernelSize,
                      float standardDeviation,
                      float high,
                      float low)
        : EdgeDetectorBase<CannyTimings>(pixel, w, h),
          m_gaussKernelSize(gaussKernelSize),
          m_standardDeviation(standardDeviation),
          m_high(high),
          m_low(low) {}
    int* getGaussKernelSize() { return &m_gaussKernelSize; }
    float* getStandardDeviation() { return &m_standardDeviation; }
    float* getHigh() { return &m_high; }
    float* getLow() { return &m_low; }

protected:
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_high = 150;
    float m_low = 100;
};

#endif//BSC_THESIS_CANNY_EDGE_DETECTOR_H
