#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_

#include <vector>
#include <CL/opencl.hpp>

class OpenCLInfo {
public:
    /*!
     * The found and stored devices
     */
    static inline std::vector<cl::Device> OPENCL_DEVICES;

    /*!
     * Finds all the opencl devices stores them in OPENCL_DEVICES and prints the
     * data for the found devices
     * \return the found devices
     */
    static std::vector<cl::Device>& GetOpenCLInfoAndDevices();
};
#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_GET_DEVICES_H_
