#include "general/main_window.h"

#ifdef CUDA_EXISTS
#include "general/cuda/gpu_info.h"
#endif

#include "general/OpenCL/get_devices.h"
#include "general/OpenCL/file_handling.h"

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
    cl::Context context(devices[0]);
    cl::Program::Sources sources;
    FileHandling::LoadOpenCLKernel("OpenCLKernels/test.cl", sources);
    std::string kernel_code2 =
        "   void kernel what(global const int* A, global const int* B, global int* C, "
        "                          global const int* N) {"
        "    simple_add(A,B,C,N);       "
        "   }";

    sources.emplace_back(kernel_code2);

    cl::Program program(context, sources);
    if (program.build(devices[0]) != CL_SUCCESS) {
        std::cout << "Error building: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                  << std::endl;
        exit(1);
    }

    // apparently OpenCL only likes arrays ...
    // N holds the number of elements in the vectors we want to add
    int N[1] = {100};
    const int n = 100;

    // create buffers on device (allocate space on GPU)
    cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * n);
    cl::Buffer buffer_N(context, CL_MEM_READ_ONLY, sizeof(int));

    // create things on here (CPU)
    int A[n], B[n];
    for (int i = 0; i < n; i++) {
        A[i] = 10;
        B[i] = n - i - 1;
    }
    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, devices[0]);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * n, B);
    queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int), N);

    // RUN ZE KERNEL
    cl::Kernel kernel(program, "what", nullptr);
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, buffer_N);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1, n));

    int C[n];
    // read result from GPU to here
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * n, C);

    std::cout << "result: {";
    for (int i : C) {
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

    return win.run();
}
