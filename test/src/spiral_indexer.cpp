#include "spiral_indexer.h"

SpiralIndexer& SpiralIndexer::operator++() {
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

SpiralIndexer SpiralIndexer::operator++(int) {
    this->operator++();
    return *this;
}
