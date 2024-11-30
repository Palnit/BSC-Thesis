#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_

#include "general/OpenCL/program.h"

namespace ClWrapper {
/*!
 * OpenCL kernel abstraction class
 */
class Kernel {
public:
    Kernel() = delete;
    /*!
     * Constructor
     * \param program The abstracted program to run the kernel on
     * \param kernelName
     */
    Kernel(ClWrapper::Program& program, const char* kernelName) : m_program(
        program), m_kernelName(kernelName) {}

    /*!
     * Starts the kernel
     * \tparam T the type of the memory
     * \tparam N the size of the memory
     * \param range the range of the kernel
     * \param memory the input memories
     * \return the time it took the kernel
     */
    template<typename... T, size_t... N>
    float operator()(cl::NDRange range, ClWrapper::Memory<T, N>& ... memory) {
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

        cl::Event event;
        m_program.m_commandQueue
            .enqueueNDRangeKernel(m_kernel,
                                  cl::NullRange,
                                  range,
                                  cl::NullRange,
                                  nullptr,
                                  &event);
        event.wait();
        auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        return (end - start) / 1000000.f;
    }
    /*!
     * Starts the kernel
     * \tparam T the type of the memory
     * \tparam N the size of the memory
     * \param range the range of the kernel
     * \param range2 the grope range of the kernel
     * \param memory the input memories
     * \return the time it took the kernel
     */
    template<typename... T, size_t... N>
    float operator()(cl::NDRange range,
                     cl::NDRange range2,
                     ClWrapper::Memory<T, N>& ... memory) {
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

        cl::Event event;
        m_program.m_commandQueue
            .enqueueNDRangeKernel(m_kernel, cl::NullRange, range, range2,
                                  nullptr, &event);
        event.wait();
        auto start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        auto end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        return (end - start) / 1000000.f;

    }

private:
    ClWrapper::Program& m_program;
    cl::Kernel m_kernel;
    std::string m_kernelName;

};
}

#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_KERNEL_H_
