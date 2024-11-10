#include "testing_window.h"
#include "general/OpenCL/get_devices.h"

int main(int argc, char* args[]) {
    TestingWindow win(
        "Testing Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1024,
        720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    OpenCLInfo::GetOpenCLInfoAndDevices();

    return win.run();
}
