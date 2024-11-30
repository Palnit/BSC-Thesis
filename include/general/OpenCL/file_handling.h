#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_FILE_HANDLING_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_FILE_HANDLING_H_

#include <string>
#include <CL/opencl.hpp>

namespace FileHandling {

/*!
 * Loads an opencl kernel into an opencl source
 * \param path the path to the kernel file
 * \param sources the opencl source
 */
void LoadOpenCLKernel(const char* path, cl::Program::Sources& sources);

}

#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_FILE_HANDLING_H_
