#include "general/OpenCL/get_devices.h"
#include "testing_window.h"

#ifdef CUDA_EXISTS
#include "general/cuda/gpu_info.h"
#endif


int main(int argc, char* args[]) {
#ifdef CUDA_EXISTS
    GetGpuInfoCuda();
#endif

    TestingWindow win(
        "Testing Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024,
        720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    OpenCLInfo::GetOpenCLInfoAndDevices();

    return win.run();
}
