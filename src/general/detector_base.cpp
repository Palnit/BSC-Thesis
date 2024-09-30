//
// Created by Palnit on 2024. 01. 14.
//

#include "include/general/detector_base.h"
#include "include/general/OpenGL_SDL/file_handling.h"

DetectorBase::DetectorBase(SDL_Surface* picture, std::string name)
    : m_base(picture), m_name(std::move(name)) {
    m_detected = SDL_CreateRGBSurface(0,
                                      m_base->w,
                                      m_base->h,
                                      m_base->format->BitsPerPixel,
                                      m_base->format->Rmask,
                                      m_base->format->Gmask,
                                      m_base->format->Bmask,
                                      m_base->format->Amask);
    SDL_BlitSurface(m_base, NULL, m_detected, NULL);

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_detected->w,
                 m_detected->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_detected->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    float verts[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 1.0f
    };

    VBO.AddElement(verts);
    EBO.AddElement({2, 3, 1, 3, 0, 1});

    vertexShader = FileHandling::LoadShader(GL_VERTEX_SHADER,
                                            "shaders/default_vertex.vert");

    fragmentShader = FileHandling::LoadShader(GL_FRAGMENT_SHADER,
                                              "shaders/default_fragment.frag");

    shaderProgram.AttachShader(vertexShader);
    shaderProgram.AttachShader(fragmentShader);

    VBO.AddAttribute({{3, 5 * sizeof(float), (void*) 0},
                      {2, 5 * sizeof(float), (void*) (3 * sizeof(float))}});
    VAO.AddVertexBuffer(VBO);
    VAO.AddElementBuffer(EBO);
}
