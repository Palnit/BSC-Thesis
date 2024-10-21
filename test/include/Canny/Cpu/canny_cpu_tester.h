#ifndef BSC_THESIS_CANNY_CPU_TESTER_H
#define BSC_THESIS_CANNY_CPU_TESTER_H
#include "Canny/canny_timings.h"
#include "tester_base.h"

class CannyCpuTester : public TesterBase {
public:
    explicit CannyCpuTester();
    void SpecializedDisplayImGui() override;
    void Test() override;

private:
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
    int m_gaussKernelSize = 3;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel;
    float* m_tangent;
    CannyTimings m_timings;
};

#endif//BSC_THESIS_CANNY_CPU_TESTER_H
