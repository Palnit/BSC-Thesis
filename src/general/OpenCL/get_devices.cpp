#include <iostream>
#include "general/OpenCL/get_devices.h"

std::vector<cl::Device> GetOpenCLInfoAndDevices() {

    if (OPENCL_DEVICES.empty()) {
        std::vector<cl::Platform> platforms;
        std::vector<cl::Device> devices_available;
        cl::Platform::get(&platforms);
        for (const auto& platform : platforms) {
            devices_available.clear();
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
            if (devices_available.empty()) continue;
            for (const auto& j : devices_available) {
                OPENCL_DEVICES.push_back(j);
            }
        }
    }

    if (OPENCL_DEVICES.empty()) {
        std::cout << "Error: There are no OpenCL devices available!"
                  << std::endl;
    } else {
        fprintf(stderr, "Found %llu OpenCl Capable device(s)\n",
                OPENCL_DEVICES.size());
    }

    for (int i = 0; i < OPENCL_DEVICES.size(); i++) {
        fprintf(stderr,
                "\nDevice %d: \"%s\"\n",
                i,
                OPENCL_DEVICES[i].getInfo<CL_DEVICE_NAME>().c_str());

        fprintf(stderr,
                "  Vendor                   :\t%s\n",
                OPENCL_DEVICES[i].getInfo<CL_DEVICE_VENDOR>().c_str());
        fprintf(stderr,
                "  Driver Version           :\t%s\n",
                OPENCL_DEVICES[i].getInfo<CL_DRIVER_VERSION>().c_str());

        auto type = OPENCL_DEVICES[i].getInfo<CL_DEVICE_TYPE>();

        if (type & CL_DEVICE_TYPE_CPU)
            fprintf(stderr,
                    "  Type                     :\tCPU\n");
        if (type & CL_DEVICE_TYPE_GPU)
            fprintf(stderr,
                    "  Type                     :\tGPU\n");
        if (type & CL_DEVICE_TYPE_ACCELERATOR)
            fprintf(stderr,
                    "  Type                     :\tAccelerator\n");
        if (type & CL_DEVICE_TYPE_DEFAULT)
            fprintf(stderr,
                    "  Type                     :\tDefault\n");

        fprintf(stderr,
                "  Max Compute Units        :\t%u\n",
                OPENCL_DEVICES[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());

        fprintf(stderr,
                "  Max Work Item Dimensions :\t%u\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>());

        auto workItemSizes = OPENCL_DEVICES[i]
            .getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        fprintf(stderr,
                "  Max Work Item Sizes      :\t%llu / %llu / %llu \n",
                workItemSizes[0], workItemSizes[1], workItemSizes[2]);

        fprintf(stderr,
                "  Max Work Group Size      :\t%llu\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

        fprintf(stderr,
                "  Max Clock Frequency      :\t%u\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());

        fprintf(stderr,
                "  Address Bits             :\t%u\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_ADDRESS_BITS>());

        fprintf(stderr,
                "  Max Mem Alloc Size       :\t%llu\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024 * 1024));

        fprintf(stderr,
                "  Global Mem Size          :\t%llu\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024));

        fprintf(stderr,
                "  Error Correction Support :\t%s\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>() == CL_TRUE
                ? "Yes"
                : "No");

        fprintf(stderr,
                "  Local Mem Type           :\t%s\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() == 1
                ? "Local"
                : "Global");

        fprintf(stderr,
                "  Local Mem Size           :\t%llu\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024);

        fprintf(stderr,
                "  Max Constant Buffer Size :\t%llu\n",
                OPENCL_DEVICES[i]
                    .getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() / 1024);

    }

    return OPENCL_DEVICES;
}
