#include "testing_imgui_display.h"
#include <imgui.h>
#include "testing_window.h"

void TestingImGuiDisplay::DisplayImGui() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(m_width / 2, m_height));
    if (!ImGui::Begin("Testing Settings", NULL,
                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking
                          | ImGuiWindowFlags_NoResize
                          | ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("Test Detector")) {
        auto* window = dynamic_cast<TestingWindow*>(m_parent);
        for (auto tester : window->m_testers) {
            tester->MainWindowDisplayImGui();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
    ImGui::SetNextWindowPos(ImVec2(m_width / 2, 0));
    ImGui::SetNextWindowSize(ImVec2(m_width / 2, m_height));
    if (!ImGui::Begin("Results", NULL,
                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking
                          | ImGuiWindowFlags_NoResize
                          | ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    auto* window = dynamic_cast<TestingWindow*>(m_parent);
    for (auto tester : window->m_testers) { tester->ResultDisplay(); }
    ImGui::End();
}
