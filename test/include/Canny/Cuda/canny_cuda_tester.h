#ifndef BSC_THESIS_TEST_INCLUDE_CANNY_CUDA_CANNY_CUDA_TESTER_H_
#define BSC_THESIS_TEST_INCLUDE_CANNY_CUDA_CANNY_CUDA_TESTER_H_

#include <vector>
#include "tester_base.h"
#include "Canny/canny_timings.h"

class CannyCudaTester : public TesterBase {
public:
    explicit CannyCudaTester();
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
};

#endif //BSC_THESIS_TEST_INCLUDE_CANNY_CUDA_CANNY_CUDA_TESTER_H_
