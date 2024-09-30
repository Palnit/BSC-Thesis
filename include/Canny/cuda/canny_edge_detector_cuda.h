//
// Created by Palnit on 2024. 01. 16.
//

#ifndef CUDA_DETECTORCUDA_H_
#define CUDA_DETECTORCUDA_H_

#include <utility>

#include "include/general/detector_base.h"
#include "GL/glew.h"
#include "include/general/OpenGL_SDL/element_buffer_object.h"
#include "include/general/OpenGL_SDL/vertex_array_object.h"
#include "include/general/OpenGL_SDL/shader_program.h"
#include "include/Canny/canny_timings.h"

/*!
 * \class CannyEdgeDetectorCuda
 * \brief Implementation of the DetectorBase class for Canny edge detection
 * on cuda
 *
 * It implements the base class and stores data related to the Canny edge
 * detection and class the cuda functions
 */
class CannyEdgeDetectorCuda : public DetectorBase {
public:

    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     */
    CannyEdgeDetectorCuda(SDL_Surface* base,
                          std::string name) : DetectorBase(
        base,
        std::move(name)) {
    }

    /*!
     * Implementation of the DetectEdge function calls from the cuda library
     * and hides all cuda related things
     */
    void DetectEdge() override;

    /*!
     * Implementation of the DisplayImGui function displays the variables
     * related to this edge detection method to be modified easily
     */
    void DisplayImGui() override;

    /*!
     * Implementation of the Display function displays the base and
     * detected image
     */
    void Display() override;
private:
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
    bool m_timingsReady = false;
    CannyTimings m_timings;

};

#endif //CUDA_DETECTORCUDA_H_
