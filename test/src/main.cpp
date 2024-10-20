
#include <iostream>
#include <random>
#include <numeric>
#include "SDL_image.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "glm/vec2.hpp"
#include "Dog/cuda/cuda_dog_edge_detection.cuh"
#include "general/cpu/gauss_blur_cpu.h"
#include "Canny/cuda/cuda_canny_edge_detection.cuh"
#include "general/cpu/morphology_cpu.h"

class SpiralIndexer {
public:
    SpiralIndexer& operator++() {
        if (2 * m_x * m_d < m_m) {
            m_x += m_d;
            return *this;
        }
        if (2 * m_y * m_d < m_m) {
            m_y += m_d;
            return *this;
        }
        m_d *= -1;
        m_m++;
        this->operator++();
        return *this;
    }

    SpiralIndexer operator++(int) {
        this->operator++();
        return *this;
    }

    int X() const { return m_x; }
    int Y() const { return m_y; }

private:
    int m_x = 0;
    int m_y = 0;
    int m_d = 1;
    int m_m = 1;
};

float DistanceOfPixels(int x1, int y1, int x2, int y2) {
    int x = (x2 - x1) * (x2 - x1);
    int y = (y2 - y1) * (y2 - y1);
    return std::sqrtf(x + y);
}

int main(int argc, char* args[]) {
    if (IMG_Init(IMG_INIT_JPG) == 0) {
        ErrorHandling::HandelSDLError("SDL IMG initialization");
        return 1;
    }

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    uint32_t format = SDL_PIXELFORMAT_ABGR8888;
#else
    uint32_t format = SDL_PIXELFORMAT_RGBA8888;
#endif

    SDL_Surface
        * loadedSurface = SDL_CreateRGBSurface(0, 100, 100, 32, 0, 0, 0, 0);

    SDL_Surface* m_base = SDL_ConvertSurfaceFormat(loadedSurface, format, 0);
    SDL_FreeSurface(loadedSurface);
    glm::vec2 a = {30, 50};
    glm::vec2 b = {80, 80};
    std::cout << m_base->w;
    for (int x = 0; x < m_base->w; ++x) {
        for (int y = 0; y < m_base->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*) m_base->pixels) + (x * 4)
                + (y * m_base->w * 4));
            color->b = 255;
        }
    }

    float dx = abs(b.x - a.x);
    int sx = a.x < b.x ? 1 : -1;
    float dy = -abs(b.y - a.y);
    int sy = a.y < b.y ? 1 : -1;
    float error = dx + dy;
    while (true) {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0, 255);
        RGBA* color = (RGBA*) (((uint8_t*) m_base->pixels) + ((int) a.x * 4)
            + ((int) a.y * m_base->w * 4));
        color->r = 255;
        color->b = 0;
        if (a.x == b.x && a.y == b.y) { break; }
        float e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            a.x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            a.y += sy;
        }
    }

    uint8_t* d_pixel2 = nullptr;

    cudaMalloc((void**) &d_pixel2,
               sizeof(uint8_t) * m_base->w
                   * m_base->h
                   * m_base->format->BytesPerPixel);

    cudaMemcpy(d_pixel2,
               m_base->pixels,
               sizeof(uint8_t) * m_base->w * m_base->h
                   * m_base->format->BytesPerPixel,
               cudaMemcpyHostToDevice);

    CudaDogDetector detector(d_pixel2,
                             m_base->w,
                             m_base->h,
                             7,
                             0.5,
                             10.0);

    auto m_detected2 = SDL_CreateRGBSurface(0,
                                            m_base->w,
                                            m_base->h,
                                            m_base->format->BitsPerPixel,
                                            m_base->format->Rmask,
                                            m_base->format->Gmask,
                                            m_base->format->Bmask,
                                            m_base->format->Amask);
    cudaMemcpy(m_detected2->pixels,
               d_pixel2,
               sizeof(uint8_t) * m_detected2->w * m_detected2->h
                   * m_detected2->format->BytesPerPixel,
               cudaMemcpyDeviceToHost);

    cudaFree(d_pixel2);

    uint8_t* d_pixel = nullptr;

    cudaMalloc((void**) &d_pixel,
               sizeof(uint8_t) * m_base->w
                   * m_base->h
                   * m_base->format->BytesPerPixel);

    cudaMemcpy(d_pixel,
               m_base->pixels,
               sizeof(uint8_t) * m_base->w * m_base->h
                   * m_base->format->BytesPerPixel,
               cudaMemcpyHostToDevice);

    CudaCannyDetector detector2(d_pixel,
                                m_base->w,
                                m_base->h,
                                5,
                                1,
                                1,
                                0.5);

    auto m_detected = SDL_CreateRGBSurface(0,
                                           m_base->w,
                                           m_base->h,
                                           m_base->format->BitsPerPixel,
                                           m_base->format->Rmask,
                                           m_base->format->Gmask,
                                           m_base->format->Bmask,
                                           m_base->format->Amask);
    cudaMemcpy(m_detected->pixels,
               d_pixel,
               sizeof(uint8_t) * m_detected->w * m_detected->h
                   * m_detected->format->BytesPerPixel,
               cudaMemcpyDeviceToHost);

    cudaFree(d_pixel);

    IMG_SavePNG(m_base, "./base.png");
    IMG_SavePNG(m_detected, "./canny_detected.png");
    IMG_SavePNG(m_detected2, "./dog_detected.png");

    IMG_SavePNG(m_detected, "./detected.png");

    SDL_Surface
        * loadedSurface2 = SDL_CreateRGBSurface(0, 100, 100, 32, 0, 0, 0, 0);

    SDL_Surface* m_base2 = SDL_ConvertSurfaceFormat(loadedSurface2, format, 0);
    SDL_FreeSurface(loadedSurface2);
    SpiralIndexer indexer;
    for (int i = 0; i < 25; i++) {
        std::cout << "x: " << indexer.X() << " Y: " << indexer.Y() << std::endl;
        indexer++;
    }
    std::vector<float> distances;

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*) m_detected2->pixels) + (x * 4)
                + (y * m_detected2->w * 4));
            if (color->r <= 20) {
                color->r = color->b = color->g = 0;
            }
        }
    }

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*) m_detected2->pixels) + (x * 4)
                + (y * m_detected2->w * 4));
            if (color->r != 0 && color->b != 0 && color->g != 0) {
                SpiralIndexer indexer2;
                bool match = false;
                for (int i = 0; i < 25; i++) {
                    int nX = x + indexer2.X();
                    int nY = y + indexer2.Y();
                    if (nX >= m_detected->w || nY >= m_detected->h) {
                        indexer2++;
                        continue;
                    }

                    RGBA* color2 =
                        (RGBA*) (((uint8_t*) m_base->pixels) + (nX * 4)
                            + (nY * m_base->w * 4));
                    if (color2->r == 255) {
                        float dis = DistanceOfPixels(x, y, nX, nY);
                        std::cout << "Dis : " << dis << std::endl;
                        distances.push_back(dis);
                        match = true;
                        break;
                    }
                    indexer2++;

                }
                if (!match) {
                    std::cout << "FUCK YOU" << std::endl;
                }
            }
        }
    }
    auto avg = std::reduce(distances.begin(), distances.end())
        / (float) distances.size();
    std::cout << "AVG : " << avg << std::endl;

    SDL_FreeSurface(m_base);
    SDL_FreeSurface(m_detected);
    SDL_FreeSurface(m_detected2);

    return 0;

}
