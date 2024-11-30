#ifndef BSC_THESIS_TESTING_WINDOW_H
#define BSC_THESIS_TESTING_WINDOW_H

#include <vector>
#include "general/OpenGL_SDL/basic_window.h"
#include "tester_base.h"
#include "testing_imgui_display.h"

class TestingWindow : public BasicWindow {
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
    TestingWindow(const char* title,
                  int x,
                  int y,
                  int width,
                  int height,
                  uint32_t flags);
    /*!
     * Destructor takes care of any data that need freeing after the program has
     * finished running
     */
    ~TestingWindow();
    /*!
     * Implementation of the RenderImGui function of the base class
     */
    void RenderImGui() override;
    /*!
     * Implementation of the Resize function of the base class
     */
    void Resize() override;
    /*!
     * Implementation of the Init function of the base class
     * \return Status
     */
    int Init() override;

    const std::vector<TesterBase*>& GetTesters() {
        return m_testers;
    }

private:
    TestingImGuiDisplay m_imGuiWindow;
    std::vector<TesterBase*> m_testers;
};

#endif//BSC_THESIS_TESTING_WINDOW_H
