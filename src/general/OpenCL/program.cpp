#include "general/OpenCL/program.h"
#include "general/OpenCL/file_handling.h"
namespace ClWrapper {
Program::Program(const cl::Device& device)
    : m_device(device),
      m_context(device),
      m_commandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE),
      m_built(false) {

}
void Program::AddSource(const char* path) {
    if (!m_built) {
        FileHandling::LoadOpenCLKernel(path, m_sources);
    }
}
bool Program::Build() {
    m_program.~Program();
    new(&m_program) cl::Program(m_context, m_sources);
    if (m_program.build(m_device) != CL_SUCCESS) {
        std::cout << "Error building: "
                  << m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device)
                  << std::endl;
        return false;
    }
    m_built = true;
    return true;
}
}