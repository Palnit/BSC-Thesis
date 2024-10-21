#include "testing_window.h"
#include "Canny/Cpu/canny_cpu_tester.h"

void TestingWindow::RenderImGui() {
    BasicWindow::RenderImGui();
    m_imGuiWindow.DisplayImGui();
}

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
    return 0;
}
TestingWindow::~TestingWindow() {
    for (auto tester : m_testers) { delete tester; }
}
