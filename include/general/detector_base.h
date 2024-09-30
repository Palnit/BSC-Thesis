//
// Created by Palint on 2024. 01. 14.
//

#ifndef DETECTORBASE_H_
#define DETECTORBASE_H_

#include <iostream>
#include <utility>
#include <SDL_surface.h>
#include <GL/glew.h>
#include "include/general/OpenGL_SDL/vertex_array_object.h"
#include "include/general/OpenGL_SDL/shader_program.h"

/*!
 * \class DetectorBase
 * \brief A base class to handel different kind of edge detector methods and
 * implementations
 *
 * It defines virtual functions that all classes that inherits need to implement
 * in order to provide a simple way to call them from an array
 */
class DetectorBase {
public:
    /*!
     * Constructor takes a picture and makes a copy of it to be used as the
     * detected picture and a name for that detector which will be used as the
     * files name latter if it implements a save method
     * \param picture The picture to be taken
     * \param name The name of the detector
     */
    DetectorBase(SDL_Surface* picture, std::string name);

    /*!
     * Destructor that frees the SDL_Surfaces of the pictures
     */
    ~DetectorBase() {
        SDL_FreeSurface(m_base);
        SDL_FreeSurface(m_detected);
    }

    /*!
     * Virtual function \n
     * Used to display the picture
     */
    virtual void Display() = 0;

    /*!
     * Virtual function \n
     * Used to detect the edge of the picture when called
     */
    virtual void DetectEdge() = 0;

    /*!
     * Virtual function \n
     * Used to display the unique settings of the detector in the base ImGui menu
     */
    virtual void DisplayImGui() = 0;

    /*!
     * Unused function right now latter used to display more than one picture
     * at a time
     * \param counter The amount to be displayed
     */
    static void SetCounter(int counter) {
        m_counter = counter;
    }
protected:
    SDL_Surface* m_base;
    SDL_Surface* m_detected;
    static inline int m_counter;
    int m_position;
    std::string m_name;
    GLuint tex;
    VertexArrayObject VAO;
    VertexBufferObject<float> VBO;
    ElementBufferObject EBO;
    GLuint vertexShader;
    GLuint fragmentShader;
    ShaderProgram shaderProgram;

};

#endif //DETECTORBASE_H_
