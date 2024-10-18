#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_PROGRAM_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_PROGRAM_H_

#include <iostream>
#include "CL/opencl.hpp"
namespace ClWrapper {
class Kernel;
template<typename T, size_t N>
class Memory;

class Program {
public:
    explicit Program(const cl::Device& device);
    void AddSource(const char* path);
    bool Build();
    bool GetBuilt() const { return m_built; }

    friend class ClWrapper::Kernel;
    template<typename T, size_t N>
    friend
    class ClWrapper::Memory;

private:
    cl::Device m_device;
    cl::Context m_context;
    cl::CommandQueue m_commandQueue;
    cl::Program m_program;
    cl::Program::Sources m_sources;
    bool m_built;
};
}
#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_PROGRAM_H_
