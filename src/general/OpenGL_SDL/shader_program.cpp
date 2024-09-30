//
// Created by Palnit on 2024. 01. 17.
//

#include "include/general/OpenGL_SDL/shader_program.h"

ShaderProgram::ShaderProgram() {
    m_program = glCreateProgram();
}

void ShaderProgram::AttachShader(GLuint shader) {
    m_shaders.push_back(shader);
    glAttachShader(m_program, shader);

}
void ShaderProgram::Bind() {
    LinkProgram();
    glUseProgram(m_program);
}
void ShaderProgram::UnBind() {
    glUseProgram(0);
}
void ShaderProgram::LinkProgram() {
    if (linked) {
        return;
    }
    linked = true;
    glLinkProgram(m_program);
}
ShaderProgram::~ShaderProgram() {
    for (auto shader : m_shaders) {
        glDetachShader(m_program, shader);
        glDeleteShader(shader);
    }
    glDeleteProgram(m_program);
}


