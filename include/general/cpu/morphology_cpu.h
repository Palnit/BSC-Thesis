#ifndef BSC_THESIS_INCLUDE_GENERAL_CPU_MORPHOLOGY_CPU_H_
#define BSC_THESIS_INCLUDE_GENERAL_CPU_MORPHOLOGY_CPU_H_

#include <cstdint>

/*!
 * Unused Morphology functions right now
 */
namespace DetectorsCPU {

void Dilation(uint8_t* image,
              uint8_t* structuringElement,
              uint8_t* output,
              int w,
              int h,
              int kernelSize);
void Erosion(uint8_t* image,
             uint8_t* structuringElement,
             uint8_t* output,
             int w,
             int h,
             int kernelSize);

}
#endif //BSC_THESIS_INCLUDE_GENERAL_CPU_MORPHOLOGY_CPU_H_
