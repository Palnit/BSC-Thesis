#ifndef BSC_THESIS_TESTING_WINDOW_H
#define BSC_THESIS_TESTING_WINDOW_H

#include <vector>
#include "general/OpenGL_SDL/basic_window.h"
#include "tester_base.h"
#include "testing_imgui_display.h"

class TestingWindow : public BasicWindow {
public:
    TestingWindow(const char* title,
                  int x,
                  int y,
                  int width,
                  int height,
                  uint32_t flags);
    ~TestingWindow();
    void RenderImGui() override;
    void Resize() override;
    int Init() override;

    const std::vector<TesterBase*>& GetTesters() {
        return m_testers;
    }

private:
    TestingImGuiDisplay m_imGuiWindow;
    std::vector<TesterBase*> m_testers;
};

#endif//BSC_THESIS_TESTING_WINDOW_H
