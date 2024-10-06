
#include <iostream>
#include <random>
#include "SDL_image.h"
#include "general/OpenGL_SDL/generic_structs.h"
#include "glm/vec2.hpp"
#include "Dog/cuda/cuda_dog_edge_detection.cuh"
#include "general/cpu/gauss_blur_cpu.h"
#include "Canny/cuda/cuda_canny_edge_detection.cuh"
#include "general/cpu/morphology_cpu.h"

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

    SDL_Surface* loadedSurface  = SDL_CreateRGBSurface(0, 100, 100, 32, 0, 0, 0, 0);

    SDL_Surface* m_base = SDL_ConvertSurfaceFormat(loadedSurface, format, 0);
    SDL_FreeSurface(loadedSurface);
    glm::vec2 a = {30,50};
    glm::vec2 b = {80,80};
    std::cout << m_base->w;
    for (int x = 0; x < m_base->w; ++x) {
        for (int y = 0; y < m_base->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*)m_base->pixels) + (x * 4)
                + (y * m_base->w * 4));
            color->b = 255;
        }
    }

    float dx = abs(b.x - a.x);
    int sx = a.x < b.x ? 1 : -1;
    float dy = -abs(b.y - a.y);
    int sy = a.y < b.y ? 1 : -1;
    float error = dx+dy;
    while(true){
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist(0,255);
        RGBA* color = (RGBA*) (((uint8_t*)m_base->pixels) + ((int)a.x * 4)
            + ((int)a.y * m_base->w * 4));
        color->r = dist(rng);
        color->g = dist(rng);
        color->b = dist(rng);
        if(a.x == b.x && a.y == b.y) {break;}
        float e2 = 2*error;
        if(e2 >= dy){
            error += dy;
            a.x += sx;
        }
        if(e2 <= dx){
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

    uint8_t skeleton[] = {0,1,0,1,1,1,0,1,0};

    auto ErosionTest = SDL_CreateRGBSurface(0,
                                           m_base->w,
                                           m_base->h,
                                           m_base->format->BitsPerPixel,
                                           m_base->format->Rmask,
                                           m_base->format->Gmask,
                                           m_base->format->Bmask,
                                           m_base->format->Amask);

    auto DilationTest = SDL_CreateRGBSurface(0,
                                            m_base->w,
                                            m_base->h,
                                            m_base->format->BitsPerPixel,
                                            m_base->format->Rmask,
                                            m_base->format->Gmask,
                                            m_base->format->Bmask,
                                            m_base->format->Amask);

    uint8_t* original =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) *m_detected->w *m_detected->h));
    uint8_t* tmp =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) *m_detected->w *m_detected->h));
    uint8_t* eroded =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) *m_detected->w *m_detected->h));
    uint8_t* finalSkell =
        static_cast<uint8_t*>(malloc(sizeof(uint8_t) *m_detected->w *m_detected->h));

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*)m_detected->pixels) + (x * 4)
                + (y * m_detected->w * 4));
            *(original + x + (y * m_detected->w )) = color->r;

            RGBA* color2 = (RGBA*) (((uint8_t*)ErosionTest->pixels) + (x * 4)
                + (y * m_detected->w * 4));
            RGBA* color3 = (RGBA*) (((uint8_t*)DilationTest->pixels) + (x * 4)
                + (y * m_detected->w * 4));
            color2->a =255;
            color3->a =255;
            (*(finalSkell + x + (y * m_detected->w ))) = 0;
        }
    }


    DetectorsCPU::Dilation(original, skeleton, tmp, m_detected->w, m_detected->h, 3);
    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            *(original + x + (y * m_detected->w )) =  *(tmp + x + (y * m_detected->w ));
        }
    }
    bool donne = false;
    while(!donne){
    DetectorsCPU::Erosion(original, skeleton, eroded, m_detected->w, m_detected->h, 3);
    DetectorsCPU::Dilation(eroded, skeleton, tmp, m_detected->w, m_detected->h, 3);



    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*)DilationTest->pixels) + (x * 4)
                + (y * m_detected->w * 4));
            color->r = color->b = color->g = (*(tmp + x + (y * m_detected->w )));
        }
    }

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            *(tmp + x + (y * m_detected->w )) = *(original + x + (y * m_detected->w )) - *(tmp + x + (y * m_detected->w ));
        }
    }

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            auto value1 =(*(finalSkell + x + (y * m_detected->w )));
            auto value2 =(*(tmp + x + (y * m_detected->w )));

            (*(finalSkell + x + (y * m_detected->w ))) =  value1 | value2;
        }
    }


    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            (*(original + x + (y * m_detected->w ))) =  (*(eroded + x + (y * m_detected->w )));
        }
    }

    int count =0;
    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            if((*(original + x + (y * m_detected->w ))) > 0){
                count++;
            }
        }
    }
    std::cout <<count;
    if(count == 0){
        donne = true;
    }
    }

    for (int x = 0; x < m_detected->w; ++x) {
        for (int y = 0; y < m_detected->h; ++y) {
            RGBA* color = (RGBA*) (((uint8_t*)ErosionTest->pixels) + (x * 4)
                + (y * m_detected->w * 4));
            color->r = color->b = color->g = (*(finalSkell + x + (y * m_detected->w )));
        }
    }

    free(original);
    free(tmp);
    free(finalSkell);
    free(eroded);

    IMG_SavePNG(m_detected, "./detected.png");
    IMG_SavePNG(ErosionTest, "./Erosion.png");
    IMG_SavePNG(DilationTest, "./Dilation.png");
    SDL_FreeSurface(m_base);
    SDL_FreeSurface(m_detected);
    SDL_FreeSurface(ErosionTest);
    SDL_FreeSurface(DilationTest);
    return 0;

}
