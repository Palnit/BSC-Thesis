#ifndef BSC_THESIS_SURFACE_PAINTERS_H
#define BSC_THESIS_SURFACE_PAINTERS_H

#include "SDL_surface.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "glm/vec2.hpp"

namespace SurfacePainters {

/*!
 * Generates an sdl surface with given color width and height
 * \param color the color of the surface
 * \param width the width
 * \param height the height
 * \return
 */
SDL_Surface* GenerateRGBSurface(RGBA color, int width, int height);
/*!
 * Draws a line on an sdl surface
 * \param surface the surface
 * \param color the color of the line
 * \param start the start coordinates
 * \param end the end coordinates
 */
void DrawLine(SDL_Surface* surface, RGBA color, glm::vec2 start, glm::vec2 end);
/*!
 * Draw a line on an sdl surface with random colors
 * \param surface the surface
 * \param start the start coordinates
 * \param end the end coordinates
 */
void DrawLine(SDL_Surface* surface, glm::vec2 start, glm::vec2 end);
/*!
 * Draws a bezier curve on a surface
 * \param surface the surface
 * \param color the color of the line
 * \param p1 p1 of bezier
 * \param p2 p2 of bezier
 * \param p3 p3 of bezier
 * \param p4 p4 of bezier
 */
void DrawCubicBezier(SDL_Surface* surface,
                     RGBA color,
                     glm::ivec2 p1,
                     glm::ivec2 p2,
                     glm::ivec2 p3,
                     glm::ivec2 p4);
/*!
 * Interpolates between two values
 * \param from from value
 * \param to to value
 * \param percent the percentage of the interpolation
 * \return the interpolated value
 */
int interpolate(int from, int to, float percent);

}// namespace SurfacePainters

#endif //BSC_THESIS_SURFACE_PAINTERS_H
