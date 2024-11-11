#include "testing_window.h"
#include "Canny/Cpu/canny_cpu_tester.h"
#include "Canny/OpenCl/canny_open_cl_tester.h"
#include "Dog/Cpu/dog_cpu_tester.h"

#ifdef CUDA_EXISTS
#include "Canny/Cuda/canny_cuda_tester.h"
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
    m_testers.push_back(new CannyCpuTester());
#ifdef CUDA_EXISTS
    m_testers.push_back(new CannyCudaTester());
#endif
    m_testers.push_back(new CannyOpenClTester());
    m_testers.push_back(new DogCpuTester());
    return 0;
}
TestingWindow::~TestingWindow() {
    for (auto tester : m_testers) { delete tester; }
}
