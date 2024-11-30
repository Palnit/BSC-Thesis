kernel void DifferenceOfGaussian(global float* kernel1,
                                     global float* kernel2,
                                     global float* finalKernel,
                                     global int* inKernelSize) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int kernelSize = *inKernelSize;
    uint index = x + (y * kernelSize);
    finalKernel[index] = kernel1[index] - kernel2[index];
}
