#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "../include/image_utils.h"

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

// Apply convolution to a single pixel
unsigned char apply_kernel(Image *img, int x, int y, int channel, float *kernel, int kernel_size) {
    float sum = 0.0f;
    int half = kernel_size / 2;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int img_x = x + kx;
            int img_y = y + ky;
            if (img_x < 0) img_x = 0;
            if (img_x >= img->width) img_x = img->width - 1;
            if (img_y < 0) img_y = 0;
            if (img_y >= img->height) img_y = img->height - 1;
            int img_index = (img_y * img->width + img_x) * img->channels + channel;
            int kernel_index = (ky + half) * kernel_size + (kx + half);
            sum += img->data[img_index] * kernel[kernel_index];
        }
    }
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

// OpenMP parallel convolution
Image* convolve_openmp(Image *input, float *kernel, int kernel_size) {
    Image *output = (Image*)malloc(sizeof(Image));
    output->width = input->width;
    output->height = input->height;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(
        input->width * input->height * input->channels
    );
    // Parallelize the outer two loops across all available threads
    #pragma omp parallel for collapse(2) schedule(dynamic)
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
    int is_blur = 0;

    if (strcmp(argv[3], "blur") == 0) {
        kernel_size = 21;      // same as serial
        float sigma = 7.0f;    // same as serial
        kernel = generate_gaussian_kernel(kernel_size, sigma);
        is_blur = 1;
    } else if (strcmp(argv[3], "edge") == 0) {
        kernel = edge_detection_3x3;
        kernel_size = 3;
    } else {
        kernel = sharpen_3x3;
        kernel_size = 3;
    }

    printf("Running OpenMP with %d threads\n", omp_get_max_threads());

    double start = omp_get_wtime();
    Image *output = convolve_openmp(input, kernel, kernel_size);
    double end = omp_get_wtime();

    printf("OpenMP convolution took: %.4f seconds\n", end - start);

    save_image(argv[2], output);

    free_image(input);
    free_image(output);
    if (is_blur) free(kernel);

    return 0;
}
