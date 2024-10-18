#include "general/main_window.h"

#ifdef CUDA_EXISTS
#include "general/cuda/gpu_info.h"
#endif

#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/file_handling.h"
#include "general/OpenCL/program.h"
#include "general/OpenCL/memory.h"
#include "general/OpenCL/kernel.h"

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

    std::vector<cl::Device> devices = GetOpenCLInfoAndDevices();
    ClWrapper::Program programTest(devices[0]);
    programTest.AddSource("OpenCLKernels/test.cl");

    ClWrapper::Memory<int, 100> A_test(programTest, CL_MEM_READ_WRITE);
    ClWrapper::Memory<int, 100> B_test(programTest, CL_MEM_READ_WRITE);
    ClWrapper::Memory<int, 100> C_test(programTest, CL_MEM_READ_WRITE);
    ClWrapper::Memory<int, 1> N_test(programTest, CL_MEM_READ_ONLY);

    N_test = 100;

    for (int i = 0; i < N_test; i++) {
        A_test[i] = 10;
        B_test[i] = 100 - i - 1;
    }

    A_test.WriteToDevice();
    B_test.WriteToDevice();
    N_test.WriteToDevice();

    ClWrapper::Kernel kernel_test(programTest, "simple_add");

    kernel_test(cl::NDRange(1, 100), A_test, B_test, C_test, N_test);

    C_test.ReadFromDevice();

    std::cout << "result2: {";
    for (int i : C_test) {
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

    return win.run();
}
