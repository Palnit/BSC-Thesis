//
// Created by Palnit on 2024. 01. 17.
//

#include "include/general/OpenGL_SDL/vertex_array_object.h"

void VertexArrayObject::Bind() const {
    glBindVertexArray(m_VAO);
}
void VertexArrayObject::UnBind() const {
    glBindVertexArray(0);
}
