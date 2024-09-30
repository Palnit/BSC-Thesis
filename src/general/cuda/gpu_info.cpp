//
// Created by Palnit on 2023. 11. 11.
//

#include "include/general/cuda/gpu_info.h"

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdio>

void GetGpuInfo() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        fprintf(stderr,
                "cudaGetDeviceCount returned %d\n-> %s\n",
                (int) error_id,
                cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0) {
        fprintf(stderr, "There are no CUDA capabile devices.\n");
        exit(EXIT_SUCCESS);
    } else {
        fprintf(stderr,
                "Found %d CUDA Capable device(s) supporting CUDA\n",
                deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, dev);

        fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        int runtimeVersion = 0;
        cudaRuntimeGetVersion(&runtimeVersion);

        fprintf(stderr,
                "  CUDA Runtime Version                    :\t%d.%d\n",
                runtimeVersion / 1000,
                (runtimeVersion % 100) / 10);

        fprintf(stderr,
                "  CUDA Compute Capability                 :\t%d.%d\n",
                deviceProp.major,
                deviceProp.minor);

        fprintf(stderr,
                "  Memory Clock Rate (MHz)                 :\t%d\n",
                deviceProp.memoryClockRate / 1024);

        fprintf(stderr,
                "  Memory Bus Width (bits)                 :\t%d\n",
                deviceProp.memoryBusWidth);

        fprintf(stderr,
                "  Peak Memory Bandwidth (GB/s)            :\t%.1f\n",
                2.0 * deviceProp.memoryClockRate
                    * ((float) deviceProp.memoryBusWidth / 8) / 1.0e6);
        fprintf(stderr,
                "  Total global memory (Gbytes)            :\t%.1f\n",
                (float) (deviceProp.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
        fprintf(stderr,
                "  Shared memory per block (Kbytes)        :\t%.1f\n",
                (float) (deviceProp.sharedMemPerBlock) / 1024.0);
        fprintf(stderr,
                "  Warp-size                               :\t%d\n",
                deviceProp.warpSize);
        fprintf(stderr,
                "  Concurrent kernels                      :\t%s\n",
                deviceProp.concurrentKernels ? "yes" : "no");
        fprintf(stderr,
                "  Concurrent computation/communication    :\t%s\n\n",
                deviceProp.deviceOverlap ? "yes" : "no");

    }
}