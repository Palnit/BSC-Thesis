#include "general/imgui_display.h"
#include <imgui.h>
#include "Canny/OpenCl/canny_edge_detector_open_cl.h"
#include "Canny/canny_detector.h"
#include "Canny/cpu/canny_edge_detector_cpu.h"
#include "Dog/cpu/dog_edge_detector_cpu.h"
#include "Dog/dog_detector.h"
#include "general/main_window.h"

#ifdef CUDA_EXISTS
#include "Canny/cuda/canny_edge_detector_cuda.cuh"
#include "Dog/cuda/dog_edge_detector_cuda.cuh"
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

    ImGui::RadioButton("DoG CPU", &m_add, 2);
#ifdef CUDA_EXISTS
    ImGui::SameLine();
    ImGui::RadioButton("DoG Cuda GPU", &m_add, 3);
#endif
    ImGui::SameLine();
    ImGui::RadioButton("DoG OpenCL GPU", &m_add, 5);

    ImGui::ListBox("Choose picture", &m_picture, VectorOfStringGetter,
                   (void*) &m_pictures, (int) m_pictures.size());

    auto* parent = dynamic_cast<MainWindow*>(m_parent);
    if (ImGui::Button("Add")) {
        std::string file = "pictures/" + m_pictures.at(m_picture);
        SDL_Surface* m_base = FileHandling::LoadImage(file.c_str());

        DetectorBase* detector;

        switch (m_add) {
            case 0:
                detector =
                    new CannyDetector<CannyEdgeDetectorCPU>(m_base, "Canny Cpu",
                                                            "canny_cpu");
                break;
#ifdef CUDA_EXISTS
            case 1:
                detector =
                    new CannyDetector<CannyEdgeDetectorCuda>(m_base,
                                                             "Canny Cuda",
                                                             "canny_cuda");
                break;
#endif
            case 2:
                detector =
                    new DogDetector<DogEdgeDetectorCPU>(m_base,
                                                        "DoG Cpu",
                                                        "dog_cpu");
                break;
#ifdef CUDA_EXISTS
            case 3:
                detector =
                    new DogDetector<DogEdgeDetectorCuda>(m_base,
                                                         "DoG Cuda",
                                                         "dog_cuda");
                break;
#endif
            case 4:
                detector =
                    new CannyDetector<CannyEdgeDetectorOpenCl>(m_base,
                                                               "Canny OpenCl",
                                                               "canny_open_cl");
                break;
            case 5:
                detector =
                    new DogDetector<DogEdgeDetectorOpenCl>(m_base, "DoG OpenCl",
                                                           "dog_open_cl");
                break;
        }

        parent->SetDetector(detector);
    }

    auto detector = parent->GetDetector();
    if (detector != nullptr) {
        detector->DisplayImGui();
    }
    ImGui::End();
}
