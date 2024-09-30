//
// Created by Palnit on 2024. 01. 20.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_
#define GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_

#include <vector>
#include "detector_base.h"
#include "include/general/OpenGL_SDL/basic_window.h"

/*!
 * \class ImGuiDisplay
 * \brief A class to display and handel the ImGui control panel of the program
 *
 * It takes care of creating and handling the detectors as well as giving the
 * created detectors to the main window for displaying
 */
class ImGuiDisplay {
public:
    /*!
     * Constructor it takes the windows current size and a pointer to the
     * main window so it can communicate with it later
     * \param width The width of the window
     * \param height The height of the window
     * \param parent The main windows it exists in
     */
    ImGuiDisplay(int width, int height, BasicWindow* parent)
        : m_width(width), m_height(height), m_parent(parent) {
    }

    /*!
     * Function to display the ImGui control panel of the program called in
     * the main window
     */
    void DisplayImGui();

    /*!
     * Getter function to be able to display a vector of elements in an ImGui
     * list box
     * \param data A pointer to the data of the object
     * \param n The number of elements in the data pointer
     * \param out_text The text that will be displayed in the list box
     * \return Notify of success
     */
    static bool VectorOfStringGetter(void* data, int n, const char** out_text) {
        const std::vector<std::string>* v = (std::vector<std::string>*) data;
        *out_text = v->at(n).c_str();
        return true;
    }

    /*!
     * A function that updates the width and height of the display are if it
     * changed
     * \param width The new width of the display
     * \param height The new height of the display
     */
    void Resize(int width, int height) {
        m_width = width;
        m_height = height;
    }
private:
    std::vector<DetectorBase*> m_detectors;
    std::vector<std::string> m_names;
    BasicWindow* m_parent;
    int m_width;
    int m_height;
    int m_add = 0;
    int m_remove = 0;
    int m_picture = 0;
    char m_buf[300] = "Name Detector";
    std::vector<std::string>
        m_pictures
        {"d2.png", "d2_90.jpg", "img.jpg", "test.jpg", "github.PNG", "d4.gif"};
};

#endif //GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_
