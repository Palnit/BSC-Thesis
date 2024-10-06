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
    cl::Platform::get(&platforms);
    for(int i=0; i<(int)platforms.size(); i++) {
        devices_available.clear();
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
        if(devices_available.size()==0) continue; // no device found in plattform i
        for(int j=0; j<(int)devices_available.size(); j++) {
            devices.push_back(devices_available[j]);
        }
    }
    if(platforms.size()==0||devices.size()==0) {
        std::cout << "Error: There are no OpenCL devices available!" << std::endl;
        return -1;
    }
    for(int i=0; i<devices.size(); i++) std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Context context(devices[0]);
    cl::Program::Sources sources;
    std::string kernel_code=
        "   void kernel simple_add(global const int* A, global const int* B, global int* C, "
        "                          global const int* N) {"
        "       int ID, Nthreads, n, ratio, start, stop;"
        ""
        "       ID = get_global_id(0);"
        "       Nthreads = get_global_size(0);"
        "       n = N[0];"
        ""
        "       ratio = (n / Nthreads);"  // number of elements for each thread
        "       start = ratio * ID;"
        "       stop  = ratio * (ID + 1);"
        ""
        "       for (int i=start; i<stop; i++)"
        "           C[i] = A[i] + B[i];"
        "   }";
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build(devices[0]) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
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
    cl::Buffer buffer_N(context, CL_MEM_READ_ONLY,  sizeof(int));

    // create things on here (CPU)
    int A[n], B[n];
    for (int i=0; i<n; i++) {
        A[i] = 10;
        B[i] = n - i - 1;
    }
    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, devices[0]);

    // push write commands to queue
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int)*n, A);
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int)*n, B);
    queue.enqueueWriteBuffer(buffer_N, CL_TRUE, 0, sizeof(int),   N);

    // RUN ZE KERNEL
    cl::Kernel kernel(program, "simple_add",nullptr);
    kernel.setArg(0,buffer_A);
    kernel.setArg(1,buffer_B);
    kernel.setArg(2,buffer_C);
    kernel.setArg(3,buffer_N);

    queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(1,n));

    int C[n];
    // read result from GPU to here
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int)*n, C);

    std::cout << "result: {";
    for (int i=0; i<n; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << "}" << std::endl;



    return win.run();
}
