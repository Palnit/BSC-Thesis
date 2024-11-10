#ifndef BSC_THESIS_CANNY_CPU_TESTER_H
#define BSC_THESIS_CANNY_CPU_TESTER_H
#include <cmath>
#include <vector>
#include "Canny/canny_timings.h"
#include "tester_base.h"

class CannyCpuTester : public TesterBase {
public:
    explicit CannyCpuTester();
    void SpecializedDisplayImGui() override;
    void ResultDisplay() override;
    void Test() override;

private:
    std::vector<CannyTimings> m_allTimings;
    std::vector<float> m_AVG;
    std::vector<int> m_missing;
    CannyTimings m_timings;
    float m_standardDeviation = 1;
    float m_highTrashHold = 1;
    float m_lowTrashHold = 0.5;
    int m_gaussKernelSize = 3;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel;
    float* m_tangent;
};

#endif//BSC_THESIS_CANNY_CPU_TESTER_H
