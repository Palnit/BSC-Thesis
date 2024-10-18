#ifndef BSC_THESIS_INCLUDE_GENERAL_OPENCL_MEMORY_H_
#define BSC_THESIS_INCLUDE_GENERAL_OPENCL_MEMORY_H_

#include "general/OpenCL/program.h"
namespace ClWrapper {

template<typename T, size_t N>
class Memory {
public:
    Memory() = delete;
    Memory(ClWrapper::Program& program, cl_mem_flags memType)
        : m_program(program),
          m_data(new T[N]),
          m_memType(memType),
          m_buffer(program.m_context, memType, sizeof(T) * N) {}

    Memory(ClWrapper::Program& program,
           T* data,
           size_t size,
           cl_mem_flags memType) : m_program(program),
                                   m_memType(memType),
                                   m_buffer(program.m_context,
                                            memType,
                                            sizeof(T) * size) {
        assert(N == size);

        m_data = new T[size];
        std::copy(data, data + size, m_data);
    }

    Memory(ClWrapper::Program& program, T (& data)[N], cl_mem_flags memType)
        : m_program(program),
          m_memType(memType),
          m_buffer(program.m_context, memType, sizeof(T) * N) {

        m_data = new T[N];
        std::copy(data, data + N, m_data);
    }

    Memory(const ClWrapper::Memory<T, N>& other)
        : m_program(other.m_program),
          m_memType(other.m_memType),
          m_buffer(other.m_buffer) {
        m_data = new T[N];
        std::copy(other.m_data, other.m_data + N, m_data);
    }

    Memory(ClWrapper::Memory<T, N>&& other) noexcept
        : m_program(other.m_program),
          m_memType(other.m_memType),
          m_buffer(std::move(other.m_buffer)) {
        m_data = other.m_data;
        other.m_data = nullptr;
    }

    Memory& operator=(const ClWrapper::Memory<T, N>& other) {
        if (this == other) {
            return *this;
        }
        m_program = other.m_program;
        m_data = new T[N];
        m_buffer = other.m_buffer;
        std::copy(other.m_data, other.m_data + N, m_data);
        return *this;
    }

    Memory& operator=(ClWrapper::Memory<T, N>&& other) noexcept {
        if (this == other) {
            return *this;
        }
        m_program = other.m_program;
        m_data = other.m_data;
        m_buffer = std::move(other.m_buffer);
        other.m_data = nullptr;
        return *this;
    }

    T& operator[](size_t position) {
        assert(N > position);
        return *(m_data + position);
    }

    ~Memory() {
        delete m_data;
    }

    void WriteToDevice() {
        m_program.m_commandQueue
            .enqueueWriteBuffer(m_buffer, CL_TRUE, 0, sizeof(T) * N, m_data);

    }

    void ReadFromDevice() {
        m_program.m_commandQueue
            .enqueueReadBuffer(m_buffer, CL_TRUE, 0, sizeof(T) * N, m_data);

    }
    friend class Kernel;

private:
    ClWrapper::Program& m_program;
    T* m_data;
    cl::Buffer m_buffer;
    cl_mem_flags m_memType;

};

template<typename T>
class Memory<T, 1> {
public:
    Memory() = delete;
    Memory(ClWrapper::Program& program, cl_mem_flags memType)
        : m_program(program),
          m_memType(memType),
          m_buffer(program.m_context, memType, sizeof(T)) {}

    Memory(ClWrapper::Program& program,
           T data,
           cl_mem_flags memType) : m_program(program),
                                   m_data(data),
                                   m_memType(memType),
                                   m_buffer(program.m_context,
                                            memType,
                                            sizeof(T)) {
    }

    Memory(const ClWrapper::Memory<T, 1>& other)
        : m_program(other.m_program),
          m_memType(other.m_memType),
          m_buffer(other.m_buffer) {
        m_data = other.m_data;
    }

    Memory(ClWrapper::Memory<T, 1>&& other) noexcept
        : m_program(other.m_program),
          m_data(std::move(other.m_data)),
          m_memType(other.m_memType),
          m_buffer(std::move(other.m_buffer)) {
    }

    Memory& operator=(const ClWrapper::Memory<T, 1>& other) {
        if (this == other) {
            return *this;
        }
        m_program = other.m_program;
        m_data = other.m_data;
        m_buffer = other.m_buffer;
        return *this;
    }

    Memory& operator=(ClWrapper::Memory<T, 1>&& other) noexcept {
        if (this == other) {
            return *this;
        }
        m_program = other.m_program;
        m_data = std::move(other.m_data);
        m_buffer = std::move(other.m_buffer);
        return *this;
    }

    Memory& operator=(T& otherData) {
        if (m_data == otherData) {
            return *this;
        }
        m_data = otherData;
        return *this;
    }

    Memory& operator=(T&& otherData) {
        if (m_data == otherData) {
            return *this;
        }
        m_data = std::move(otherData);
        return *this;
    }

    void WriteToDevice() {
        m_program.m_commandQueue
            .enqueueWriteBuffer(m_buffer, CL_TRUE, 0, sizeof(T), &m_data);

    }

    void ReadFromDevice() {
        m_program.m_commandQueue
            .enqueueReadBuffer(m_buffer, CL_TRUE, 0, sizeof(T), &m_data);

    }

    friend class Kernel;
private:
    ClWrapper::Program& m_program;
    T m_data;
    cl::Buffer m_buffer;
    cl_mem_flags m_memType;

};

}

#endif //BSC_THESIS_INCLUDE_GENERAL_OPENCL_MEMORY_H_
