#ifndef BSC_THESIS_CANNY_EDGE_DETECTOR_CPU_H
#define BSC_THESIS_CANNY_EDGE_DETECTOR_CPU_H

#include "Canny/canny_edge_detector.h"
class CannyDetectorCPU;

class CannyEdgeDetectorCPU : public CannyEdgeDetector {
public:
    CannyEdgeDetectorCPU() = default;
    std::shared_ptr<uint8_t> Detect() override;
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
}// namespace DetectorsCPU

#endif//BSC_THESIS_CANNY_EDGE_DETECTOR_CPU_H
