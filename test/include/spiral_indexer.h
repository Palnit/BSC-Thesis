#ifndef BSC_THESIS_SPIRAL_INDEXER_H
#define BSC_THESIS_SPIRAL_INDEXER_H

class SpiralIndexer {
public:

    SpiralIndexer& operator++();
    SpiralIndexer operator++(int);
    int X() const { return m_x; }
    int Y() const { return m_y; }

private:
    int m_x = 0;
    int m_y = 0;
    int m_d = 1;
    int m_m = 1;
};


#endif //BSC_THESIS_SPIRAL_INDEXER_H
