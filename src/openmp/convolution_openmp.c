#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "../include/image_utils.h"

// Define convolution kernels
float gaussian_blur_10x10[100] = {
    1, 4, 7, 10, 12, 12, 10, 7, 4, 1,
    4, 16, 26, 36, 44, 44, 36, 26, 16, 4,
    7, 26, 41, 56, 68, 68, 56, 41, 26, 7,
    10, 36, 56, 76, 92, 92, 76, 56, 36, 10,
    12, 44, 68, 92, 112, 112, 92, 68, 44, 12,
    12, 44, 68, 92, 112, 112, 92, 68, 44, 12,
    10, 36, 56, 76, 92, 92, 76, 56, 36, 10,
    7, 26, 41, 56, 68, 68, 56, 41, 26, 7,
    4, 16, 26, 36, 44, 44, 36, 26, 16, 4,
    1, 4, 7, 10, 12, 12, 10, 7, 4, 1
};
float edge_detection_3x3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
float sharpen_3x3[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

void normalize_kernel(float *kernel, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size * size; i++) sum += kernel[i];
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

unsigned char apply_kernel(Image *img, int x, int y, int channel, float *kernel, int kernel_size) {
    float sum = 0.0;
    int half_size = kernel_size / 2;
    for (int ky = -half_size; ky <= half_size; ky++) {
        for (int kx = -half_size; kx <= half_size; kx++) {
            int img_x = x + kx;
            int img_y = y + ky;
            if (img_x < 0) img_x = 0;
            if (img_x >= img->width) img_x = img->width - 1;
            if (img_y < 0) img_y = 0;
            if (img_y >= img->height) img_y = img->height - 1;
            int img_index = (img_y * img->width + img_x) * img->channels + channel;
            int kernel_index = (ky + half_size) * kernel_size + (kx + half_size);
            sum += img->data[img_index] * kernel[kernel_index];
        }
    }
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

Image* convolve_openmp(Image *input, float *kernel, int kernel_size) {
    Image *output = (Image*)malloc(sizeof(Image));
    output->width = input->width;
    output->height = input->height;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(
        input->width * input->height * input->channels
    );
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            for (int c = 0; c < input->channels; c++) {
                int index = (y * input->width + x) * input->channels + c;
                output->data[index] = apply_kernel(input, x, y, c, kernel, kernel_size);
            }
        }
    }
    return output;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <output_image> <filter_type>\n", argv[0]);
        printf("filter_type: blur, edge, sharpen\n");
        return 1;
    }
    Image *input = load_image(argv[1]);
    if (!input) return 1;
    float *kernel;
    int kernel_size;
    int blur_iterations = 1;
    if (strcmp(argv[3], "blur") == 0) {
        normalize_kernel(gaussian_blur_10x10, 10);
        kernel = gaussian_blur_10x10;
        kernel_size = 10;
        blur_iterations = 10; // Apply blur 10 times for maximum effect
    } else if (strcmp(argv[3], "edge") == 0) {
        kernel = edge_detection_3x3;
        kernel_size = 3;
    } else {
        kernel = sharpen_3x3;
        kernel_size = 3;
    }
    clock_t start = clock();
    Image *output = input;
    for (int i = 0; i < blur_iterations; i++) {
        Image *temp = convolve_openmp(output, kernel, kernel_size);
        if (output != input) free_image(output);
        output = temp;
    }
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("OpenMP convolution took: %.4f seconds\n", time_taken);
    save_image(argv[2], output);
    free_image(input);
    if (output != input) free_image(output);
    return 0;
}
