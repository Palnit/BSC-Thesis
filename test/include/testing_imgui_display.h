#ifndef BSC_THESIS_TESTING_IMGUI_DISPLAY_H
#define BSC_THESIS_TESTING_IMGUI_DISPLAY_H

#include "general/OpenGL_SDL/basic_window.h"
class TestingImGuiDisplay {
public:
    TestingImGuiDisplay(int width, int height, BasicWindow* parent)
        : m_width(width),
          m_height(height),
          m_parent(parent) {}

    void DisplayImGui();

    void Resize(int width, int height) {
        m_width = width;
        m_height = height;
    }

private:
    int m_width;
    int m_height;
    BasicWindow* m_parent;
};

#endif//BSC_THESIS_TESTING_IMGUI_DISPLAY_H
