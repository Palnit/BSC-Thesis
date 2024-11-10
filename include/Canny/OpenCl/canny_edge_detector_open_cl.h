#ifndef BSC_THESIS_INCLUDE_CANNY_OPENCL_CANNYEDGEDETECTOROPENCL_H_
#define BSC_THESIS_INCLUDE_CANNY_OPENCL_CANNYEDGEDETECTOROPENCL_H_

#include "general/detector_base.h"
#include "GL/glew.h"
#include "general/OpenGL_SDL/element_buffer_object.h"
#include "general/OpenGL_SDL/vertex_array_object.h"
#include "general/OpenGL_SDL/shader_program.h"
#include "Canny/canny_timings.h"

class CannyEdgeDetectorOpenCl : public DetectorBase {
public:
    CannyEdgeDetectorOpenCl(SDL_Surface* base,
                            std::string name) : DetectorBase(
        base,
        std::move(name)) {
    }

    void DetectEdge() override;
    void DisplayImGui() override;
    void Display() override;
private:
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
    bool m_timingsReady = false;
    CannyTimings m_timings;

};

#endif //BSC_THESIS_INCLUDE_CANNY_OPENCL_CANNYEDGEDETECTOROPENCL_H_
