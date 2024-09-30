//
// Created by Palnit on 2023. 11. 11.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_

#include "include/general/OpenGL_SDL/basic_window.h"
#include "include/general/detector_base.h"
#include "include/Canny/cuda/canny_edge_detector_cuda.h"
#include "include/general/OpenGL_SDL/file_handling.h"
#include "imgui_display.h"

/*!
 * \class MainWindow
 * \brief The main display class of the program
 *
 * It inherits from the BasicWindow class and implements some of it's virtual
 * function needed for this project and has function to handel if a detector has
 * been created or destroyed and takes care of the detectors lifetime
 */
class MainWindow : public BasicWindow {

public:
    /*!
     * Constructor for the class same as the basic windows constructor
     * \param title The title of the window
     * \param x The horizontal position of the window
     * \param y The vertical position of the window
     * \param width The width of the window
     * \param height The height of the window
     * \param flags Flags for the sdl window creation function SDL_WINDOW_OPENGL
     * is always appended
     */
    MainWindow(const char* title,
                           int x,
                           int y,
                           int w,
                           int h,
                           Uint32 flags) : BasicWindow(title,
                                                       x,
                                                       y,
                                                       w,
                                                       h,
                                                       flags),
                                           m_display(m_width, m_height, this) {
    }

    /*!
     * Destructor takes care of any data that need freeing after the program has
     * finished running
     */
    ~MainWindow();

    /*!
     * Implementation of the Init function of the base class
     * \return Status
     */
    int Init() override;

    /*!
     * Implementation of the Render function of the base class
     */
    void Render() override;

    /*!
     * Implementation of the RenderImGui function of the base class
     */
    void RenderImGui() override;

    /*!
     * Adds a detector to the classes internal storage and displays the picture
     * the detector will use
     * \param Detector The detector
     */
    void AddDetector(DetectorBase* Detector);

    /*!
     * Removes a detector from the classes internal storage
     * \param Detector The detector to be removed
     */
    void RemoveDetector(DetectorBase* Detector);

    /*!
     * Implementation of the Resize function of the base class
     */
    void Resize() override;

private:
    std::vector<DetectorBase*> m_detectors;
    CannyEdgeDetectorCuda* m_det;
    ImGuiDisplay m_display;

};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
