#include "include/general//main_window.h"
#include "include/general/cuda/gpu_info.h"

int main(int argc, char* args[]) {
    GetGpuInfo();

    MainWindow win("Edge Detector",
                   SDL_WINDOWPOS_CENTERED,
                   SDL_WINDOWPOS_CENTERED,
                   1024,
                   720,
                   SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
                       | SDL_WINDOW_RESIZABLE);

    return win.run();
}
