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
    /*!
     * Constructor
     * \param device the opencl device this program is for
     */
    explicit Program(const cl::Device& device);
    /*!
     * Add an opencl kernel file to this program
     * \param path the path to the kernel file
     */
    void AddSource(const char* path);
    /*!
     * build this opencl program
     * \return the check if its failed
     */
    bool Build();
    /*!
     * Gets if this is built or not
     * \return the bool check
     */
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
