#include "tester_base.h"

#include <utility>

void TesterBase::MainWindowDisplayImGui() {
    if (ImGui::BeginTabItem(m_name.c_str())) {
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
        if (ImGui::Button("Start Test", {ImGui::GetWindowWidth(), 50})) {
            Test();
        }
        ImGui::EndTabItem();
    }
}
TesterBase::TesterBase(const std::string& name)
    : m_name(name),
      m_height(100),
      m_width(100),
      m_iterations(100),
      m_normalLines(5),
      m_bezierLines(5) {
    m_backGroundColor[0] = 0;
    m_backGroundColor[1] = 0;
    m_backGroundColor[2] = 255;
    m_backGroundColor[3] = 255;
    m_linesColor[0] = 255;
    m_linesColor[1] = 0;
    m_linesColor[2] = 0;
    m_linesColor[3] = 255;
}
