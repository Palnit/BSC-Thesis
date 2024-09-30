//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_

#include "include/general/detector_base.h"
#include "include/Canny/canny_timings.h"

/*!
 * \class CannyEdgeDetectorCPU
 * \brief Implementation of the DetectorBase class for Canny edge detection
 * on cpu
 *
 * It implements the base class and stores data related to the dog edge detection
 */
class CannyEdgeDetectorCPU : public DetectorBase {
public:

    /*!
     * Implementation of the base constructor
     * \param picture The picture to be taken
     * \param name The name of the detector
     */
    CannyEdgeDetectorCPU(SDL_Surface* base,
                         std::string name) : DetectorBase(base,
                                                          std::move(name)),
                                             m_w(m_base->w),
                                             m_h(m_base->h) {

    }

    /*!
     * Implementation of the DetectEdge function class the detection functions
     */
    void DetectEdge() override;

    /*!
     * Implementation of the DisplayImGui function displays the variables
     * related to this edge detection method to be modified easily
     */
    void DisplayImGui() override;

    /*!
     * Implementation of the Display function displays the base and
     * detected image
     */
    void Display() override;
private:
    int m_w;
    int m_h;
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
    bool m_timingsReady = false;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel;
    float* m_tangent;
    CannyTimings m_timings;
};

/*!
 * A namespace containing the implementation of the cpu implementation of the
 * edge detection algorithm
 */
namespace DetectorsCPU {

/*!
 * This function uses the sobel operator to calculate the gradient and tangent
 * of the picture at every pixel
 * \param src The source grey scaled image
 * \param dest The output image
 * \param tangent The tangent of the image
 * \param w The width of the image
 * \param h The height of the image
 */
void DetectionOperator(float* src, float* dest, float* tangent, int w, int h);

/*!
 * This function keeps the current pixel value if it's the maximum gradient in
 * the tangent direction
 * \param src The source gradients
 * \param dest The destination
 * \param tangent The tangent at every pixel
 * \param w The width of the image
 * \param h The height of the image
 */
void NonMaximumSuppression(float* src,
                           float* dest,
                           float* tangent,
                           int w,
                           int h);

/*!
 * This function defines strong and week edges based on 2 arbitrary thresholds
 * \param src The source gradients
 * \param dest The destination
 * \param w The width of the image
 * \param h The height of the image
 * \param high The high threshold
 * \param low The low threshold
 */
void DoubleThreshold(float* src,
                     float* dest,
                     int w,
                     int h,
                     float high,
                     float low);

/*!
 * This function keeps the week edges if they have at least one strong edge
 * adjacent to them
 * \param src The source image
 * \param dest The destination
 * \param w The width of the image
 * \param h The height of the image
 */
void Hysteresis(float* src, float* dest, int w, int h);
}
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_
