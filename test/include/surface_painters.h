#ifndef BSC_THESIS_SURFACE_PAINTERS_H
#define BSC_THESIS_SURFACE_PAINTERS_H

#include "SDL_surface.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "glm/vec2.hpp"

namespace SurfacePainters {

SDL_Surface* GenerateRGBSurface(RGBA color, int width, int height);
void DrawLine(SDL_Surface* surface, RGBA color, glm::vec2 start, glm::vec2 end);
void DrawLine(SDL_Surface* surface, glm::vec2 start, glm::vec2 end);
void DrawCubicBezier(SDL_Surface* surface,
                     RGBA color,
                     glm::ivec2 p1,
                     glm::ivec2 p2,
                     glm::ivec2 p3,
                     glm::ivec2 p4);
int interpolate(int from, int to, float percent);

}// namespace SurfacePainters

#endif//BSC_THESIS_SURFACE_PAINTERS_H
