#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include image utilities as C
extern "C" {
    #include "../include/image_utils.h"
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Generate a true normalized Gaussian kernel (size must be odd)
float* generate_gaussian_kernel(int size, float sigma) {
    float *kernel = (float*)malloc(size * size * sizeof(float));
    int half = size / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
    return kernel;
}

float edge_detection_3x3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
float sharpen_3x3[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

// CUDA kernel: each thread processes one pixel's one channel
__global__ void convolution_kernel(unsigned char *input, unsigned char *output,
                                    int width, int height, int channels,
                                    float *kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = kernel_size / 2;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int ky = -half; ky <= half; ky++) {
            for (int kx = -half; kx <= half; kx++) {
                int img_x = x + kx;
                int img_y = y + ky;
                // Clamp to edges
                if (img_x < 0) img_x = 0;
                if (img_x >= width) img_x = width - 1;
                if (img_y < 0) img_y = 0;
                if (img_y >= height) img_y = height - 1;

                int img_index = (img_y * width + img_x) * channels + c;
                int kernel_index = (ky + half) * kernel_size + (kx + half);
                sum += input[img_index] * kernel[kernel_index];
            }
        }
        if (sum < 0.0f) sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;
        int out_index = (y * width + x) * channels + c;
        output[out_index] = (unsigned char)sum;
    }
}

// Main CUDA convolution function
Image* convolve_cuda(Image *input, float *kern, int kernel_size) {
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t kern_size = kernel_size * kernel_size * sizeof(float);

    // Allocate output on host
    Image *output = (Image*)malloc(sizeof(Image));
    output->width = width;
    output->height = height;
    output->channels = channels;
    output->data = (unsigned char*)malloc(img_size);

    // Allocate device memory
    unsigned char *d_input, *d_output;
    float *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kern_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, input->data, img_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kern, kern_size, cudaMemcpyHostToDevice));

    // Launch kernel with 16x16 thread blocks
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    convolution_kernel<<<gridDim, blockDim>>>(d_input, d_output,
                                               width, height, channels,
                                               d_kernel, kernel_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output->data, d_output, img_size, cudaMemcpyDeviceToHost));

    // Cleanup device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <output_image> <filter_type>\n", argv[0]);
        printf("filter_type: blur, edge, sharpen\n");
        return 1;
    }

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);

    // Load image
    Image *input = load_image(argv[1]);
    if (!input) return 1;

    // Select kernel
    float *kern;
    int kernel_size;
    int is_blur = 0;

    if (strcmp(argv[3], "blur") == 0) {
        kernel_size = 21;      // same as serial
        float sigma = 7.0f;    // same as serial
        kern = generate_gaussian_kernel(kernel_size, sigma);
        is_blur = 1;
    } else if (strcmp(argv[3], "edge") == 0) {
        kern = edge_detection_3x3;
        kernel_size = 3;
    } else {
        kern = sharpen_3x3;
        kernel_size = 3;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Image *output = convolve_cuda(input, kern, kernel_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("CUDA convolution took: %.4f seconds\n", milliseconds / 1000.0f);

    save_image(argv[2], output);

    free_image(input);
    free_image(output);
    if (is_blur) free(kern);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
