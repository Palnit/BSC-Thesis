typedef struct RGBA {
    uchar r;
    uchar g;
    uchar b;
    uchar a;
} RGBA;

kernel void ConvertToGreyScale(global const uchar* src, global float* out, global const int* iw, global const int* ih) {
    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);
    int w = *iw;
    int h = *ih;

    if (x >= w || y >= h) {
        return;
    }

    RGBA* color = (RGBA*) (src + (x * 4) + (y * w * 4));
    *(out + x + (y * w)) = (0.299 * color->r)
        + (0.587 * color->g)
        + (0.114 * color->b);
}

kernel void CopyBack(global const float* src, global uchar* out, global const int* iw, global const int* ih){
    uint x = get_local_id(0) + get_group_id(0) * get_local_size(0);
    uint y = get_local_id(1) + get_group_id(1) * get_local_size(1);

    int w = *iw;
    int h = *ih;

    if (x >= w || y >= h) {
        return;
    }

    RGBA* color = (RGBA*) (out + (x * 4) + (y * w * 4));
    color->r = color->g = color->b = *(src + x + (y * w));
}

global float sumG = 0;
kernel void GetGaussian(global float* out,global int* inKernelSize,global float* inSigma){
    int x = get_global_id(0);
    int y = get_global_id(1);
    float sigma = *inSigma;
    int kernelSize = *inKernelSize;

    int index = x + (y * kernelSize);

    int k = (kernelSize - 1) / 2;

    float xp = (((x + 1.f) - (1.f + k)) * ((x + 1.f) - (1.f + k)));
    float yp = (((y + 1.f) - (1.f + k)) * ((y + 1.f) - (1.f + k)));
    out[index] =
        (1.f / (2.f * M_PI_F * sigma * sigma))
            * exp(-((xp + yp) / (2.f * sigma * sigma)));
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (index == 0) {
        sumG = 0;

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                int nIndex = j + (i * kernelSize);
                sumG += out[nIndex];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    out[index] = out[index] / sumG;
}


kernel void GaussianFilter(global float* src,
                           global float* dest,
                           global float* gauss,
                           global int* inKernelSize,global const int* iw, global const int* ih){
    int kernelSize = *inKernelSize;

    int w = *iw;
    int h = *ih;

    uint col = get_local_id(0) + get_group_id(0) * (get_local_size(0) - kernelSize);
    uint row = get_local_id(1) + get_group_id(1) * (get_local_size(1) - kernelSize);
    int k = (kernelSize - 1) / 2;
    uint col_i = col - k;
    uint row_i = row - k;



    local float src_local[32][32];
    uint index = col_i + (row_i * w);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_local[get_local_id(0)][get_local_id(1)] = src[index];
    } else {
        src_local[get_local_id(0)][get_local_id(1)] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    float sum = 0;


    if (get_local_id(0) > k - 1 && get_local_id(1) > k - 1 && get_local_id(0) < 32 - k
        && get_local_id(1) < 32 - k && col_i < w && row_i < h) {

        for (int i = -k; i <= k; i++) {
            for (int j = -k; j <= k; j++) {
                uint gaussIndex = (i + k) + ((j + k) * kernelSize);
                sum = fma(src_local[get_local_id(0) + i][get_local_id(1) + j],
                           gauss[gaussIndex],
                           sum);
            }
        }
        dest[index] = sum;
    }


}
