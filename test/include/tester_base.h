#ifndef BSC_THESIS_TESTER_BASE_H
#define BSC_THESIS_TESTER_BASE_H

#include <string>
#include "general/OpenGL_SDL/generic_structs.h"
#include "imgui.h"
#include "general/timings_base.h"
#include <cmath>
#include <vector>
#include <filesystem>

class TesterBase {
public:
    explicit TesterBase(const std::string& name,
                        const std::string& internalName)
        : m_name(name),
          m_internalName("./" + internalName),
          m_height(100),
          m_width(100),
          m_iterations(100),
          m_normalLines(5),
          m_bezierLines(5), m_selected(false) {
        m_backGroundColor[0] = 0;
        m_backGroundColor[1] = 0;
        m_backGroundColor[2] = 255;
        m_backGroundColor[3] = 255;
        m_linesColor[0] = 255;
        m_linesColor[1] = 0;
        m_linesColor[2] = 0;
        m_linesColor[3] = 255;
    }

    void MainWindowDisplayImGui() {
        m_selected = ImGui::BeginTabItem(m_name.c_str());
        if (m_selected) {
            ImGui::InputInt("Iteration Count", &m_iterations);
            ImGui::SeparatorText("Pictures");
            ImGui::InputInt("Test Pictures Height", &m_height);
            ImGui::InputInt("Test Pictures Width", &m_width);
            ImGui::DragInt4("Back Ground Color", m_backGroundColor, 1, 0, 255);
            ImGui::SeparatorText("Lines");
            ImGui::InputInt("Normal Lines Count", &m_normalLines);
            ImGui::InputInt("Bezier Lines count", &m_bezierLines);
            ImGui::DragInt4("Lines Color", m_linesColor, 1, 0, 255);
            SpecializedDisplayImGui();
            ImGui::SetCursorPos({0, ImGui::GetWindowHeight() - 60});
            if (ImGui::Button("Start Test",
                              {ImGui::GetWindowWidth() / 2, 50})) {
                Test();
            }
            ImGui::SameLine();
            if (ImGui::Button("Save Test Data",
                              {ImGui::GetWindowWidth() / 2, 50})) {
                SaveData();
            }

            ImGui::EndTabItem();
        }
    }
    virtual void ResultDisplay() = 0;
    virtual void SpecializedDisplayImGui() = 0;
    virtual void Test() = 0;
    virtual void SaveData() = 0;
    float DistanceOfPixels(int x1, int y1, int x2, int y2) {
        int x = (x2 - x1) * (x2 - x1);
        int y = (y2 - y1) * (y2 - y1);
        return std::sqrtf(x + y);
    }

protected:
    std::vector<float> m_AVG;
    std::vector<int> m_missing;
    std::string m_name;
    std::string m_internalName;
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
