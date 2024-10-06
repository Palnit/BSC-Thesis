#include "general//main_window.h"
#include <CL/cl2.hpp>

#ifdef CUDA_EXISTS
#include "general/cuda/gpu_info.h"
#endif

int main(int argc, char* args[]) {

#ifdef CUDA_EXISTS
    GetGpuInfo();
#endif

    MainWindow win("Edge Detector",
                   SDL_WINDOWPOS_CENTERED,
                   SDL_WINDOWPOS_CENTERED,
                   1024,
                   720,
                   SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
                       | SDL_WINDOW_RESIZABLE);

    std::vector<cl::Device> devices;
    std::vector<cl::Platform> platforms; // get all platforms
    std::vector<cl::Device> devices_available;
    int n = 0; // number of available devices
    cl::Platform::get(&platforms);
    for(int i=0; i<(int)platforms.size(); i++) {
        devices_available.clear();
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
        if(devices_available.size()==0) continue; // no device found in plattform i
        for(int j=0; j<(int)devices_available.size(); j++) {
            n++;
            devices.push_back(devices_available[j]);
        }
    }
    if(platforms.size()==0||devices.size()==0) {
        std::cout << "Error: There are no OpenCL devices available!" << std::endl;
        return -1;
    }
    for(int i=0; i<n; i++) std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;



    return win.run();
}
