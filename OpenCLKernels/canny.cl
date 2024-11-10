kernel void DetectionOperator(global float* src,
                                  global float* gradient,
                                  global float* tangent,
                                  global int* iw,
                                  global int* ih) {
    int w = *iw;
    int h = *ih;

    uint col = get_local_id(0) + get_group_id(0) * (get_local_size(0) - 3);
    uint row = get_local_id(1) + get_group_id(1) * (get_local_size(1) - 3);
    uint col_i = col - 1;
    uint row_i = row - 1;

    local float src_local[32][32];
    uint index = col_i + (row_i * w);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_local[get_local_id(0)][get_local_id(1)] = src[index];
    } else {
        src_local[get_local_id(0)][get_local_id(1)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};

    float SumX = 0;
    float SumY = 0;

    if (get_local_id(0) > 1 - 1 && get_local_id(1) > 1 - 1 && get_local_id(0) < 32 - 1
        && get_local_id(1) < 32 - 1 && col_i < w && row_i < h) {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
            uint SumIndex =(i + 1) + ((j + 1) * 3);
                SumX = fma(src_local[get_local_id(0) + i][get_local_id(1) + j],
                            SobelX[SumIndex],
                            SumX);
                SumY = fma(src_local[get_local_id(0) + i][get_local_id(1) + j],
                            SobelY[SumIndex],
                            SumY);
            }
        }
        gradient[index] = hypot(SumX, SumY);
        float angle = (atan2(SumX, SumY) * 180.f) / M_PI_F;
        if (angle < 0) {
            angle += 180;
        }
        tangent[index] = angle;
    }

}


kernel void NonMaximumSuppression( global float* gradient_in,
                                      global float* gradient_out,
                                      global float* tangent,
                                      global int* iw,
                                      global int* ih) {

    int w = *iw;
    int h = *ih;

    uint col = get_local_id(0) + get_group_id(0) * (get_local_size(0) - 3);
    uint row = get_local_id(1) + get_group_id(1) * (get_local_size(1) - 3);
    uint col_i = col - 1;
    uint row_i = row - 1;

    local float src_local[32][32];
    uint index = col_i + (row_i * w);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_local[get_local_id(0)][get_local_id(1)] = gradient_in[index];
    } else {
        src_local[get_local_id(0)][get_local_id(1)] = 2000;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(0) <= 1 - 1 || get_local_id(1) <= 1 - 1 || get_local_id(0) >= 32 - 1
        || get_local_id(1) >= 32 - 1 || col_i >= w || row_i >= h) {
        return;
    }

    float tangentA = tangent[index];
    float gradientA = src_local[get_local_id(0)][get_local_id(1)];
    float gradientP = 2000;
    float gradientN = 2000;

    if ((0 <= tangentA && tangentA < 22.5)
        || (157.5 <= tangentA && tangentA <= 180)) {
        gradientP = src_local[get_local_id(0)][get_local_id(1) + 1];
        gradientN = src_local[get_local_id(0)][get_local_id(1) - 1];
    } else if (22.5 <= tangentA && tangentA < 67.5) {
        gradientP = src_local[get_local_id(0) + 1][get_local_id(1) - 1];
        gradientN = src_local[get_local_id(0) - 1][get_local_id(1) + 1];
    } else if (67.5 <= tangentA && tangentA < 112.5) {
        gradientP = src_local[get_local_id(0) + 1][get_local_id(1)];
        gradientN = src_local[get_local_id(0) - 1][get_local_id(1)];
    } else if (112.5 <= tangentA && tangentA < 157.5) {
        gradientP = src_local[get_local_id(0) - 1][get_local_id(1) - 1];
        gradientN = src_local[get_local_id(0) + 1][get_local_id(1) + 1];
    }

    if (gradientA < gradientN || gradientA < gradientP) {
        gradientA = 0.f;
    }

    gradient_out[index] = gradientA;
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


kernel void Hysteresis(global float* gradient_in,
                           global float* gradient_out,
                           global int* iw,
                           global int* ih) {
    int w = *iw;
    int h = *ih;

    uint col = get_local_id(0) + get_group_id(0) * (get_local_size(0) - 3);
    uint row = get_local_id(1) + get_group_id(1) * (get_local_size(1) - 3);
    uint col_i = col - 1;
    uint row_i = row - 1;

    local float src_local[32][32];
    uint index = col_i + (row_i * w);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_local[get_local_id(0)][get_local_id(1)] = gradient_in[index];
    } else {
        src_local[get_local_id(0)][get_local_id(1)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (get_local_id(0) <= 1 - 1 || get_local_id(1) <= 1 - 1 || get_local_id(0) >= 32 - 1
        || get_local_id(1) >= 32 - 1 || col_i >= w || row_i >= h) {
        return;
    }


    bool strong = false;

    float gradientA = src_local[get_local_id(0)][get_local_id(1)];

    gradient_out[index] = gradientA;
    if (gradientA != 125.f) {
        return;
    }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            if (src_local[get_local_id(0) + j][get_local_id(1) + j] == 255.f) {
                strong = true;
            }
        }
    }
    if (strong) {
        gradientA = 255.f;
    } else {
        gradientA = 0.f;
    }

    gradient_out[index] = gradientA;
}
