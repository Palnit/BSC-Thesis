#ifndef BSC_THESIS_EDGE_DETECTOR_BASE_H
#define BSC_THESIS_EDGE_DETECTOR_BASE_H

#include <cstdint>
#include <memory>
#include <type_traits>
#include "general/timings_base.h"

template<class T>
class EdgeDetectorBase {
public:
    static_assert(std::is_base_of<TimingsBase, T>::value,
                  "Template type must have a base type of TimingsBase");

    EdgeDetectorBase() = default;
    EdgeDetectorBase(uint8_t* pixels, int w, int h)
        : m_pixels(pixels),
          m_w(w),
          m_h(h) {}

    virtual std::shared_ptr<uint8_t> Detect() = 0;
    T GetTimings() const { return m_timings; }

    int getW() const { return m_w; }
    int getH() const { return m_h; }

    void setPixels(uint8_t* pixels) { m_pixels = pixels; }
    void setW(int w) { m_w = w; }
    void setH(int h) { m_h = h; }
    void setStride(int stride) { m_stride = stride; }

protected:
    uint8_t* m_pixels;
    uint8_t* m_detected;
    int m_w;
    int m_h;
    int m_stride;
    T m_timings;
};

#endif//BSC_THESIS_EDGE_DETECTOR_BASE_H
