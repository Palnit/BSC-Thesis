#ifndef BSC_THESIS_TESTER_BASE_H
#define BSC_THESIS_TESTER_BASE_H

#include <string>
#include "general/OpenGL_SDL/generic_structs.h"
#include "imgui.h"
class TesterBase {

public:
    explicit TesterBase(const std::string& mName);
    void MainWindowDisplayImGui();
    virtual void ResultDisplay() = 0;
    virtual void SpecializedDisplayImGui() = 0;
    virtual void Test() = 0;

protected:
    std::string m_name;
    int m_backGroundColor[4];
    int m_linesColor[4];
    int m_width;
    int m_height;
    int m_iterations;
    int m_normalLines;
    int m_bezierLines;
};

#endif//BSC_THESIS_TESTER_BASE_H
