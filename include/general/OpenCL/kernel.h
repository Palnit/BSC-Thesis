#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_

#include "general/OpenCL/program.h"

namespace ClWrapper {
class Kernel {
public:
    Kernel() = delete;
    Kernel(ClWrapper::Program& program, const char* kernelName) : m_program(
        program), m_kernelName(kernelName) {}

    template<typename... T, size_t... N>
    void operator()(cl::NDRange range, ClWrapper::Memory<T, N>& ... memory) {
        if (!m_program.GetBuilt()) {
            m_program.Build();
        }
        m_kernel.~Kernel();
        new(&m_kernel) cl::Kernel(m_program.m_program, m_kernelName);
        size_t i = 0;
        (
            [&] {
                m_kernel.setArg(i, memory.m_buffer);
                i++;
            }(),
            ...);

        m_program.m_commandQueue
            .enqueueNDRangeKernel(m_kernel, cl::NullRange, range);
    }
private:
    ClWrapper::Program& m_program;
    cl::Kernel m_kernel;
    std::string m_kernelName;

};
}

#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_
