
#include <iostream>
#include <random>
#include <numeric>
#include "SDL_image.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "glm/vec2.hpp"
#include "general/cpu/gauss_blur_cpu.h"
#include "general/cpu/morphology_cpu.h"
#include "surface_painters.h"
#include "spiral_indexer.h"


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

    SDL_Surface* m_base = SurfacePainters::GenerateRGBSurface({0,0,255,255},100,100);
    glm::vec2 a = {30, 50};
    glm::vec2 b = {80, 80};
    std::cout << m_base->w;

    SurfacePainters::DrawCubicBezier(m_base,{255,0,0,255},{0,100},{0,0},{100,100},{100,0});

//    uint8_t* d_pixel2 = nullptr;
//
//    cudaMalloc((void**) &d_pixel2,
//               sizeof(uint8_t) * m_base->w
//                   * m_base->h
//                   * m_base->format->BytesPerPixel);
//
//    cudaMemcpy(d_pixel2,
//               m_base->pixels,
//               sizeof(uint8_t) * m_base->w * m_base->h
//                   * m_base->format->BytesPerPixel,
//               cudaMemcpyHostToDevice);
//
//    CudaDogDetector detector(d_pixel2,
//                             m_base->w,
//                             m_base->h,
//                             7,
//                             0.5,
//                             10.0);
//
//    auto m_detected2 = SDL_CreateRGBSurface(0,
//                                            m_base->w,
//                                            m_base->h,
//                                            m_base->format->BitsPerPixel,
//                                            m_base->format->Rmask,
//                                            m_base->format->Gmask,
//                                            m_base->format->Bmask,
//                                            m_base->format->Amask);
//    cudaMemcpy(m_detected2->pixels,
//               d_pixel2,
//               sizeof(uint8_t) * m_detected2->w * m_detected2->h
//                   * m_detected2->format->BytesPerPixel,
//               cudaMemcpyDeviceToHost);
//
//    cudaFree(d_pixel2);
//
//    uint8_t* d_pixel = nullptr;
//
//    cudaMalloc((void**) &d_pixel,
//               sizeof(uint8_t) * m_base->w
//                   * m_base->h
//                   * m_base->format->BytesPerPixel);
//
//    cudaMemcpy(d_pixel,
//               m_base->pixels,
//               sizeof(uint8_t) * m_base->w * m_base->h
//                   * m_base->format->BytesPerPixel,
//               cudaMemcpyHostToDevice);
//
//    CudaCannyDetector detector2(d_pixel,
//                                m_base->w,
//                                m_base->h,
//                                5,
//                                1,
//                                1,
//                                0.5);
//
//    auto m_detected = SDL_CreateRGBSurface(0,
//                                           m_base->w,
//                                           m_base->h,
//                                           m_base->format->BitsPerPixel,
//                                           m_base->format->Rmask,
//                                           m_base->format->Gmask,
//                                           m_base->format->Bmask,
//                                           m_base->format->Amask);
//    cudaMemcpy(m_detected->pixels,
//               d_pixel,
//               sizeof(uint8_t) * m_detected->w * m_detected->h
//                   * m_detected->format->BytesPerPixel,
//               cudaMemcpyDeviceToHost);
//
//    cudaFree(d_pixel);
//
    IMG_SavePNG(m_base, "./base.png");
//    IMG_SavePNG(m_detected, "./canny_detected.png");
//    IMG_SavePNG(m_detected2, "./dog_detected.png");
//
//    IMG_SavePNG(m_detected, "./detected.png");

    SpiralIndexer indexer;
    for (int i = 0; i < 25; i++) {
        std::cout << "x: " << indexer.X() << " Y: " << indexer.Y() << std::endl;
        indexer++;
    }
    std::vector<float> distances;

//    for (int x = 0; x < m_detected->w; ++x) {
//        for (int y = 0; y < m_detected->h; ++y) {
//            RGBA* color = (RGBA*) (((uint8_t*) m_detected2->pixels) + (x * 4)
//                + (y * m_detected2->w * 4));
//            if (color->r <= 20) {
//                color->r = color->b = color->g = 0;
//            }
//        }
//    }
//
//    for (int x = 0; x < m_detected->w; ++x) {
//        for (int y = 0; y < m_detected->h; ++y) {
//            RGBA* color = (RGBA*) (((uint8_t*) m_detected2->pixels) + (x * 4)
//                + (y * m_detected2->w * 4));
//            if (color->r != 0 && color->b != 0 && color->g != 0) {
//                SpiralIndexer indexer2;
//                bool match = false;
//                for (int i = 0; i < 25; i++) {
//                    int nX = x + indexer2.X();
//                    int nY = y + indexer2.Y();
//                    if (nX >= m_detected->w || nY >= m_detected->h) {
//                        indexer2++;
//                        continue;
//                    }
//
//                    RGBA* color2 =
//                        (RGBA*) (((uint8_t*) m_base->pixels) + (nX * 4)
//                            + (nY * m_base->w * 4));
//                    if (color2->r == 255) {
//                        float dis = DistanceOfPixels(x, y, nX, nY);
//                        std::cout << "Dis : " << dis << std::endl;
//                        distances.push_back(dis);
//                        match = true;
//                        break;
//                    }
//                    indexer2++;
//
//                }
//                if (!match) {
//                    std::cout << "FUCK YOU" << std::endl;
//                }
//            }
//        }
//    }
    auto avg = std::reduce(distances.begin(), distances.end())
        / (float) distances.size();
    std::cout << "AVG : " << avg << std::endl;

    SDL_FreeSurface(m_base);
//    SDL_FreeSurface(m_detected);
//    SDL_FreeSurface(m_detected2);

    return 0;

}
