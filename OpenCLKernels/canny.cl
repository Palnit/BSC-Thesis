kernel void DetectionOperator(global float* src,
                                  global float* dest,
                                  global float* tangent,
                                  global int* iw,
                                  global int* ih) {
    int w = *iw;
    int h = *ih;

    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    if (x >= w || y >= h) {
        return;
    }

    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};

    float SumX = 0;
    float SumY = 0;

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int ix = x + i;
            int jx = y + j;

            if (ix < 0) { ix = 0; }
            if (ix >= w) { ix = w - 1; }
            if (jx < 0) { jx = 0; }
            if (jx >= h) { jx = h - 1; }
            uint index = ix + (jx * w);
            uint index2 = (i + 1) + ((j + 1) * 3);
            SumX = fma(src[index],
                             SobelX[index2] , SumX);
            SumY = fma(src[index],
                             SobelY[index2], SumY);
        }
    }
    *(dest + x + (y * w)) = hypot(SumX, SumY);
    float angle = (atan2(SumX, SumY) * 180.f) / M_PI_F;
    if (angle < 0) {
        angle += 180;
    }
    uint index = x + (y * w);
    tangent[index]  = angle;


}


kernel void NonMaximumSuppression( global float* src,
                                      global float* dest,
                                      global float* tangent,
                                      global int* iw,
                                      global int* ih) {

    int w = *iw;
    int h = *ih;

    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    uint index = x + (y * w);
    if (x >= w || y >= h) {
        return;
    }


    int yp;
    int yn;
    int xp;
    int xn;

    float tangentA = tangent[index];
    float gradientA = src[index];
    float gradientP = 2000;
    float gradientN = 2000;

    yp = y + 1;
    if (yp >= h) { yp = h - 1; }
    xp = x + 1;
    if (xp >= h) { xp = h - 1; }
    yn = y - 1;
    if (yn < 0) { yn = 0; }
    xn = x - 1;
    if (xn < 0) { xn = 0; }

    if ((0 <= tangentA && tangentA < 22.5)
        || (157.5 <= tangentA && tangentA <= 180)) {
        uint indexP = x + (yp * w);
        uint indexN =x + (yn * w);
        gradientP = src[indexP];
        gradientN = src[indexN];
    } else if (22.5 <= tangentA && tangentA < 67.5) {
        uint indexP = xp + (yn * w);
        uint indexN =xn + (yp * w);
        gradientP = src[indexP];
        gradientN = src[indexN];
    } else if (67.5 <= tangentA && tangentA < 112.5) {
        uint indexP = xp + (y * w);
        uint indexN =xn + (y * w);
        gradientP = src[indexP];
        gradientN = src[indexN];
    } else if (112.5 <= tangentA && tangentA < 157.5) {
        uint indexP = xn + (yn * w);
        uint indexN = xp + (yp * w);
        gradientP = src[indexP];
        gradientN = src[indexN];
    }

    if (gradientA < gradientN || gradientA < gradientP) {
        gradientA = 0.f;
    }
    dest[index] = gradientA;
}
kernel void DoubleThreshold(global float* gradient_in,
                                global float* gradient_out,
                                global int* iw,
                                global int* ih,
                                global float* ihigh,
                                global float* ilow) {
    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);
    int w = *iw;
    int h = *ih;
    float high = *ihigh;
    float low = *ilow;

    if (x >= w || y >= h) {
        return;
    }

    uint index = x + (y * w);

    if (gradient_in[index] >= high) {
        gradient_out[index] = 255.f;
    } else if (gradient_in[index] < high
        && gradient_in[index] >= low) {
        gradient_out[index] = 125.f;
    } else {
        gradient_out[index] = 0.f;
    }
}


kernel void Hysteresis(global float* src,
                           global float* dest,
                           global int* iw,
                           global int* ih) {
    int w = *iw;
    int h = *ih;

    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    uint index = x + (y * w);
    if (x >= w || y >= h) {
        return;
    }

    bool strong = false;

    dest[index] = src[index];
    if (dest[index] != 125.f) { return; }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int ix = x + i;
            int jy = y + j;
            if (ix < 0) { ix = 0; }
            if (ix >= w) { ix = w - 1; }
            if (jy < 0) { jy = 0; }
            if (jy >= h) { jy = h - 1; }
            uint index2 = ix + (jy * w);

            if (src[index2] == 255.f) { strong = true; }
        }
    }
    if (strong) {
         dest[index] = 255.f;
    } else {
        dest[index] = 0;
    }
}
