#include "surface_painters.h"

#include <random>

namespace SurfacePainters {
SDL_Surface* GenerateRGBSurface(RGBA color, int width, int height) {
#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    uint32_t format = SDL_PIXELFORMAT_ABGR8888;
#else
    uint32_t format = SDL_PIXELFORMAT_RGBA8888;
#endif

    SDL_Surface* loadedSurface =
        SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);

    SDL_Surface* surface = SDL_ConvertSurfaceFormat(loadedSurface, format, 0);
    SDL_FreeSurface(loadedSurface);
    for (int x = 0; x < surface->w; ++x) {
        for (int y = 0; y < surface->h; ++y) {
            RGBA* pixels = (RGBA*) (((uint8_t*) surface->pixels) + (x * 4)
                                    + (y * surface->w * 4));
            *pixels = color;
        }
    }
    return surface;
}

void DrawLine(SDL_Surface* surface,
              RGBA color,
              glm::vec2 start,
              glm::vec2 end) {
    float dx = abs(end.x - start.x);
    int sx = start.x < end.x ? 1 : -1;
    float dy = -abs(end.y - start.y);
    int sy = start.y < end.y ? 1 : -1;
    float error = dx + dy;
    while (true) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
        RGBA* pixel =
            (RGBA*) (((uint8_t*) surface->pixels) + ((int) start.x * 4)
                     + ((int) start.y * surface->w * 4));
        *pixel = color;
        if (start.x == end.x && start.y == end.y) { break; }
        float e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            start.x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            start.y += sy;
        }
    }
}

void DrawLine(SDL_Surface* surface, glm::vec2 start, glm::vec2 end) {
    float dx = abs(end.x - start.x);
    int sx = start.x < end.x ? 1 : -1;
    float dy = -abs(end.y - start.y);
    int sy = start.y < end.y ? 1 : -1;
    float error = dx + dy;
    while (true) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
        RGBA* pixel =
            (RGBA*) (((uint8_t*) surface->pixels) + ((int) start.x * 4)
                     + ((int) start.y * surface->w * 4));
        pixel->r = dist(rng);
        pixel->g = dist(rng);
        pixel->b = dist(rng);
        pixel->a = 255;
        if (start.x == end.x && start.y == end.y) { break; }
        float e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            start.x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            start.y += sy;
        }
    }
}

void DrawCubicBezier(SDL_Surface* surface,
                     RGBA color,
                     glm::ivec2 p1,
                     glm::ivec2 p2,
                     glm::ivec2 p3,
                     glm::ivec2 p4) {

    for (float i = 0; i < 1; i += 0.0001) {
        int xa = interpolate(p1.x, p2.x, i);
        int ya = interpolate(p1.y, p2.y, i);
        int xb = interpolate(p2.x, p3.x, i);
        int yb = interpolate(p2.y, p3.y, i);
        int xc = interpolate(p3.x, p4.x, i);
        int yc = interpolate(p3.y, p4.y, i);

        int xm = interpolate(xa, xb, i);
        int ym = interpolate(ya, yb, i);
        int xn = interpolate(xb, xc, i);
        int yn = interpolate(yb, yc, i);

        int x = interpolate(xm, xn, i);
        int y = interpolate(ym, yn, i);

        RGBA* pixel = (RGBA*) (((uint8_t*) surface->pixels) + ((int) x * 4)
                               + ((int) y * surface->w * 4));

        *pixel = color;
    }
}

int interpolate(int from, int to, float percent) {
    int difference = to - from;
    return from + (difference * percent);
}
}// namespace SurfacePainters
