#ifndef DOG_OPEN_CL_TESTER_H
#define DOG_OPEN_CL_TESTER_H

#include <vector>
#include "Dog/dog_timings.h"
#include "tester_base.h"

class DogOpenClTester : public TesterBase {
public:
    explicit DogOpenClTester();
    void ResultDisplay() override;
    void SpecializedDisplayImGui() override;
    void Test() override;

private:
    std::vector<DogTimings> m_allTimings;
    std::vector<float> m_AVG;
    std::vector<int> m_missing;
    DogTimings m_timings;
    int m_gaussKernelSize = 7;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 0.7;
    float m_threshold = 20;
};

#endif//DOG_OPEN_CL_TESTER_H
