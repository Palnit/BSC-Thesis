#ifndef BSC_THESIS_DOG_CPU_TESTER_H
#define BSC_THESIS_DOG_CPU_TESTER_H

#include <vector>
#include "Dog/dog_timings.h"
#include "tester_base.h"

class DogCpuTester : public TesterBase {
public:
    explicit DogCpuTester();
    void ResultDisplay() override;
    void SpecializedDisplayImGui() override;
    void Test() override;

private:
    std::vector<DogTimings> m_allTimings;
    std::vector<float> m_AVG;
    std::vector<int> m_missing;
    DogTimings m_timings;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel1;
    float* m_kernel2;
    float* m_finalKernel;
    int m_gaussKernelSize = 7;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 0.7;
    float m_threshold = 20;
    bool m_timingsReady = false;
};

#endif//BSC_THESIS_DOG_CPU_TESTER_H
