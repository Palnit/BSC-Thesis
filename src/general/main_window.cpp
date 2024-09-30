//
// Created by Palnit on 2023. 11. 11.
//

#include "include/general//main_window.h"
#include "include/general/detector_base.h"
#include "include/general/OpenGL_SDL/file_handling.h"
#include "include/general/OpenGL_SDL/generic_structs.h"

#include <ctime>
#include <algorithm>

#include <cuda_runtime.h>

#include <implot.h>

int MainWindow::Init() {
    return 0;
}

void MainWindow::Render() {
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, m_width, m_height);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    for (DetectorBase* detector : m_detectors) {
        detector->Display();
    }
}

void MainWindow::RenderImGui() {
    bool t = true;
    ImGui::ShowMetricsWindow(&t);
    ImGui::ShowDemoWindow(&t);
    m_display.DisplayImGui();

}
MainWindow::~MainWindow() {
    for (DetectorBase* detector : m_detectors) {
        free(detector);
    }
}
void MainWindow::AddDetector(DetectorBase* Detector) {
    m_detectors.push_back(Detector);
}
void MainWindow::RemoveDetector(DetectorBase* Detector) {
    auto position = std::find(m_detectors.begin(), m_detectors.end(), Detector);
    if (position != m_detectors.end()) {
        free(*position);
        m_detectors.erase(position);
    }
}
void MainWindow::Resize() {
    BasicWindow::Resize();
    m_display.Resize(m_width, m_height);
}
