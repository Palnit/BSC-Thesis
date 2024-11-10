#ifndef BSC_THESIS_TESTER_BASE_H
#define BSC_THESIS_TESTER_BASE_H

#include <string>
#include "general/OpenGL_SDL/generic_structs.h"
#include "imgui.h"
#include <cmath>

class TesterBase {

public:
    explicit TesterBase(const std::string& mName);
    void MainWindowDisplayImGui();
    virtual void ResultDisplay() = 0;
    virtual void SpecializedDisplayImGui() = 0;
    virtual void Test() = 0;
    float DistanceOfPixels(int x1, int y1, int x2, int y2) {
        int x = (x2 - x1) * (x2 - x1);
        int y = (y2 - y1) * (y2 - y1);
        return std::sqrtf(x + y);
    }

protected:
    std::string m_name;
    int m_backGroundColor[4];
    int m_linesColor[4];
    int m_width;
    int m_height;
    int m_iterations;
    int m_normalLines;
    int m_bezierLines;
    bool m_selected;
};

#endif//BSC_THESIS_TESTER_BASE_H
