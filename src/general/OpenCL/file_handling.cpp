#include <iostream>
#include <fstream>
#include <sstream>
#include "general/OpenCL/file_handling.h"

namespace FileHandling {
void LoadOpenCLKernel(const char* path,
                      cl::Program::Sources& sources) {
    std::string kernelCode;
    std::ifstream kernelFile;
    kernelFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        kernelFile.open(path);
        std::stringstream kernelStream;
        kernelStream << kernelFile.rdbuf();
        kernelFile.close();
        kernelCode = kernelStream.str();
    } catch (std::ifstream::failure e) {
        std::cout << "Error" << std::endl;
    }
    sources.emplace_back(kernelCode);
}
}

