#ifndef BSC_THESIS_TESTING_IMGUI_DISPLAY_H
#define BSC_THESIS_TESTING_IMGUI_DISPLAY_H

#include "general/OpenGL_SDL/basic_window.h"
class TestingImGuiDisplay {
public:
    /*!
     * Constructor it takes the windows current size and a pointer to the
     * main window so it can communicate with it later
     * \param width The width of the window
     * \param height The height of the window
     * \param parent The main windows it exists in
     */
    TestingImGuiDisplay(int width, int height, BasicWindow* parent)
        : m_width(width),
          m_height(height),
          m_parent(parent) {}

    /*!
     * Function to display the ImGui control panel of the program called in
     * the main window
     */
    void DisplayImGui();

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
    int m_width;
    int m_height;
    BasicWindow* m_parent;
};

#endif//BSC_THESIS_TESTING_IMGUI_DISPLAY_H
