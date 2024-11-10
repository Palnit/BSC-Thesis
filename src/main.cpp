#include "general/main_window.h"
#include "general/OpenCL/get_devices.h"

#ifdef CUDA_EXISTS
#include "general/cuda/gpu_info.h"
#endif

#include <map>
int main(int argc, char* args[]) {

#ifdef CUDA_EXISTS
    GetGpuInfoCuda();
#endif

    MainWindow win("Edge Detector",
                   SDL_WINDOWPOS_CENTERED,
                   SDL_WINDOWPOS_CENTERED,
                   1024,
                   720,
                   SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
                       | SDL_WINDOW_RESIZABLE);

    OpenCLInfo::GetOpenCLInfoAndDevices();

    return win.run();
}
