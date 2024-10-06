#include "general/cpu/morphology_cpu.h"

void DetectorsCPU::Dilation(uint8_t* image,
                            uint8_t* structuringElement,
                            uint8_t* output,
                            int w,
                            int h,
                            int kernelSize) {

    int k = (kernelSize - 1) / 2;

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            float max = 0;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    int ix = x + i;
                    int jx = y + j;
                    if (ix < 0) {
                        ix = 0;
                    }
                    if (ix >= w) {
                        ix = w - 1;
                    }
                    if (jx < 0) {
                        jx = 0;
                    }
                    if (jx >= h) {
                        jx = h - 1;
                    }

                    if(*(structuringElement+(i+k)+((j+k)+kernelSize)) == 1)
                    {
                        float value = *(image+ix+(jx*w));
                        if( value > max){
                            max = value;
                        }
                    }
                }
            }
            *(output + x + (y * w)) = max;
        }
    }

}
void DetectorsCPU::Erosion(uint8_t* image,
                           uint8_t* structuringElement,
                           uint8_t* output,
                           int w,
                           int h,
                           int kernelSize) {

    int k = (kernelSize - 1) / 2;

    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            float min = 255;
            for (int i = -k; i <= k; i++) {
                for (int j = -k; j <= k; j++) {
                    int ix = x + i;
                    int jx = y + j;
                    if (ix < 0) {
                        ix = 0;
                    }
                    if (ix >= w) {
                        ix = w - 1;
                    }
                    if (jx < 0) {
                        jx = 0;
                    }
                    if (jx >= h) {
                        jx = h - 1;
                    }

                    if(*(structuringElement+(i+k)+((j+k)+kernelSize)) == 1)
                    {
                        float value = *(image+ix+(jx*w));
                        if( value < min){
                            min = value;
                        }
                    }
                }
            }
            *(output + x + (y * w)) = min;
        }
    }
}
