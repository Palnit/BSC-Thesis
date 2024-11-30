#include "testing_window.h"

#include "Dog/dog_tester.h"
#include "Dog/cpu/dog_edge_detector_cpu.h"
#include "Canny/canny_tester.h"
#include "Canny/cpu/canny_edge_detector_cpu.h"

#ifdef CUDA_EXISTS
#include "Dog/cuda/dog_edge_detector_cuda.cuh"
#include "Canny/cuda/canny_edge_detector_cuda.cuh"
#endif

void TestingWindow::RenderImGui() { m_imGuiWindow.DisplayImGui(); }

TestingWindow::TestingWindow(const char* title,
                             int x,
                             int y,
                             int width,
                             int height,
                             uint32_t flags)
    : BasicWindow(title, x, y, width, height, flags),
      m_imGuiWindow(m_width, m_height, this) {}
void TestingWindow::Resize() {
    BasicWindow::Resize();
    m_imGuiWindow.Resize(m_width, m_height);
}
int TestingWindow::Init() {
    m_testers
        .push_back(new CannyTester<CannyEdgeDetectorCPU>("Canny Cpu",
                                                         "canny_cpu"));
#ifdef CUDA_EXISTS
    m_testers
        .push_back(new CannyTester<CannyEdgeDetectorCuda>("Canny Cuda",
                                                          "canny_cuda"));
#endif
    m_testers.push_back(new CannyTester<CannyEdgeDetectorOpenCl>("Canny OpenCl",
                                                                 "canny_open_cl"));

    m_testers
        .push_back(new DogTester<DogEdgeDetectorCPU>("DoG Cpu", "dog_cpu"));
#ifdef CUDA_EXISTS
    m_testers
        .push_back(new DogTester<DogEdgeDetectorCuda>("DoG Cuda", "dog_cuda"));
#endif
    m_testers.push_back(new DogTester<DogEdgeDetectorOpenCl>("DoG OpenCl",
                                                             "dog_open_cl"));
    return 0;
}
TestingWindow::~TestingWindow() {
    for (auto tester : m_testers) { delete tester; }
}
