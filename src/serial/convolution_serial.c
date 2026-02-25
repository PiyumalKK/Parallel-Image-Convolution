// 10x10 Gaussian kernel (approximate, not normalized yet)
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
float gaussian_blur_7x7[49] = {
    0.000789, 0.0066, 0.0228, 0.0351, 0.0228, 0.0066, 0.000789,
    0.0066,   0.055,  0.191,  0.294,  0.191,  0.055,  0.0066,
    0.0228,   0.191,  0.631,  0.972,  0.631,  0.191,  0.0228,
    0.0351,   0.294,  0.972,  1.5,    0.972,  0.294,  0.0351,
    0.0228,   0.191,  0.631,  0.972,  0.631,  0.191,  0.0228,
    0.0066,   0.055,  0.191,  0.294,  0.191,  0.055,  0.0066,
    0.000789, 0.0066, 0.0228, 0.0351, 0.0228, 0.0066, 0.000789
};
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "../include/image_utils.h"

// Define convolution kernels
float gaussian_blur_3x3[9] = {
    1.0/16, 2.0/16, 1.0/16,
    2.0/16, 4.0/16, 2.0/16,
    1.0/16, 2.0/16, 1.0/16
};

float gaussian_blur_5x5[25] = {
    1,  4,  6,  4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1,  4,  6,  4, 1
};

// Normalize kernel
void normalize_kernel(float *kernel, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size * size; i++) sum += kernel[i];
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}
float edge_detection_3x3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
float sharpen_3x3[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

// Apply convolution to single pixel
unsigned char apply_kernel(Image *img, int x, int y, int channel, float *kernel, int kernel_size) {
    float sum = 0.0;
    int half_size = kernel_size / 2;
    for (int ky = -half_size; ky <= half_size; ky++) {
        for (int kx = -half_size; kx <= half_size; kx++) {
            int img_x = x + kx;
            int img_y = y + ky;
            // Handle borders (clamp to edge)
            if (img_x < 0) img_x = 0;
            if (img_x >= img->width) img_x = img->width - 1;
            if (img_y < 0) img_y = 0;
            if (img_y >= img->height) img_y = img->height - 1;
            int img_index = (img_y * img->width + img_x) * img->channels + channel;
            int kernel_index = (ky + half_size) * kernel_size + (kx + half_size);
            sum += img->data[img_index] * kernel[kernel_index];
        }
    }
    // Clamp result to [0, 255]
    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

// Main serial convolution function
Image* convolve_serial(Image *input, float *kernel, int kernel_size) {
    Image *output = (Image*)malloc(sizeof(Image));
    output->width = input->width;
    output->height = input->height;
    output->channels = input->channels;
    output->data = (unsigned char*)malloc(
        input->width * input->height * input->channels
    );
    // Process each pixel
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
    // Load image
    Image *input = load_image(argv[1]);
    if (!input) return 1;
    // Select kernel
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
    // Perform convolution and measure time
    clock_t start = clock();
    Image *output = input;
    for (int i = 0; i < blur_iterations; i++) {
        Image *temp = convolve_serial(output, kernel, kernel_size);
        if (i > 0) free_image(output);
        output = temp;
    }
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Serial convolution took: %.4f seconds\n", time_taken);
    // Save result
    save_image(argv[2], output);
    // Cleanup
    free_image(input);
    if (output != input) free_image(output);
    return 0;
}
