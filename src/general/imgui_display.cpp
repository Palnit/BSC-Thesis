#include "general/imgui_display.h"
#include <imgui.h>
#include "Canny/OpenCl/canny_edge_detector_open_cl.h"
#include "Canny/canny_detector.h"
#include "Canny/cpu/canny_edge_detector_cpu.h"
#include "Dog/cpu/dog_edge_detector_cpu.h"
#include "Dog/dog_detector.h"
#include "general/main_window.h"

#ifdef CUDA_EXISTS
#include "Canny/cuda/canny_detector_cuda.h"
#include "Dog/cuda/dog_detector_cuda.h"
#endif

void ImGuiDisplay::DisplayImGui() {
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(m_width / 2.5f, m_height));
    if (!ImGui::Begin("Edge Detector Options", NULL,
                      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking
                          | ImGuiWindowFlags_NoResize)) {
        ImGui::End();
        return;
    }
    ImGui::RadioButton("Canny CPU", &m_add, 0);
#ifdef CUDA_EXISTS
    ImGui::SameLine();
    ImGui::RadioButton("Canny Cuda GPU", &m_add, 1);
#endif
    ImGui::SameLine();
    ImGui::RadioButton("Canny OpenCL GPU", &m_add, 4);

    ImGui::RadioButton("DOG CPU", &m_add, 2);
#ifdef CUDA_EXISTS
    ImGui::SameLine();
    ImGui::RadioButton("DOG Cuda GPU", &m_add, 3);
#endif
    ImGui::SameLine();
    ImGui::RadioButton("DOG OpenCL GPU", &m_add, 5);

    ImGui::InputText("Detector Name", m_buf, 300);

    ImGui::ListBox("Choose picture", &m_picture, VectorOfStringGetter,
                   (void*) &m_pictures, (int) m_pictures.size());

    if (ImGui::Button("Add")) {
        auto* parent = dynamic_cast<MainWindow*>(m_parent);
        std::string file = "pictures/" + m_pictures.at(m_picture);
        SDL_Surface* m_base = FileHandling::LoadImage(file.c_str());

        DetectorBase* detector;

        switch (m_add) {
            case 0:
                detector =
                    new CannyDetector<CannyEdgeDetectorCPU>(m_base, m_buf);
                break;
#ifdef CUDA_EXISTS
            case 1:
                detector = new CannyEdgeDetectorCuda(m_base, m_buf);
                break;
#endif
            case 2:
                detector = new DogDetector<DogEdgeDetectorCPU>(m_base, m_buf);
                break;
#ifdef CUDA_EXISTS
            case 3:
                detector = new DogEdgeDetectorCuda(m_base, m_buf);
                break;
#endif
            case 4:
                detector =
                    new CannyDetector<CannyEdgeDetectorOpenCl>(m_base, m_buf);
                break;
            case 5:
                detector =
                    new DogDetector<DogEdgeDetectorOpenCl>(m_base, m_buf);
                break;
        }

        m_detectors.push_back(detector);
        parent->AddDetector(m_detectors.back());
        m_names.emplace_back(m_buf);
    }

    ImGui::ListBox("Detectors", &m_remove, VectorOfStringGetter,
                   (void*) &m_names, (int) m_names.size());

    if (ImGui::Button("Remove")) {
        auto* parent = dynamic_cast<MainWindow*>(m_parent);
        parent->RemoveDetector(m_detectors.at(m_remove));
        m_detectors.erase(m_detectors.begin() + m_remove);
        m_names.erase(m_names.begin() + m_remove);
    }
    ImGui::Separator();
    if (ImGui::BeginTabBar("Detector Options")) {
        for (DetectorBase* detector : m_detectors) { detector->DisplayImGui(); }
        ImGui::EndTabBar();
    }
    ImGui::End();
}
