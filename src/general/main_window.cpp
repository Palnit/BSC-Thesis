//
// Created by Palnit on 2023. 11. 11.
//

#include "general//main_window.h"
#include "general/detector_base.h"
#include "general/OpenGL_SDL/file_handling.h"
#include "general/OpenGL_SDL/generic_structs.h"

#include <ctime>
#include <algorithm>

#include <implot.h>

int MainWindow::Init() {
    return 0;
}

void MainWindow::Render() {
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, m_width, m_height);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    if (m_detector != nullptr) {
        m_detector->Display();
    }
}

void MainWindow::RenderImGui() {
    m_display.DisplayImGui();

}
MainWindow::~MainWindow() {
    if (m_detector != nullptr) {
        delete m_detector;
    }
}
void MainWindow::SetDetector(DetectorBase* Detector) {
    if (m_detector != nullptr) {
        delete m_detector;
    }
    m_detector = Detector;
}

void MainWindow::Resize() {
    BasicWindow::Resize();
    m_display.Resize(m_width, m_height);
}
DetectorBase* MainWindow::GetDetector() {
    return m_detector;
}
