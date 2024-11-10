#ifndef BSC_THESIS_TEST_INCLUDE_CANNY_OPENCL_CANNY_OPEN_CL_TESTER_H_
#define BSC_THESIS_TEST_INCLUDE_CANNY_OPENCL_CANNY_OPEN_CL_TESTER_H_

#include <vector>
#include "tester_base.h"
#include "Canny/canny_timings.h"

class CannyOpenClTester : public TesterBase {
public:
    explicit CannyOpenClTester();
    void ResultDisplay() override;
    void SpecializedDisplayImGui() override;
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
};

#endif //BSC_THESIS_TEST_INCLUDE_CANNY_OPENCL_CANNY_OPEN_CL_TESTER_H_
