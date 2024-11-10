#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_

#include <vector>
#include <CL/opencl.hpp>

class OpenCLInfo {
public:
    static inline std::vector<cl::Device> OPENCL_DEVICES;

    static std::vector<cl::Device>& GetOpenCLInfoAndDevices();
};
#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_
