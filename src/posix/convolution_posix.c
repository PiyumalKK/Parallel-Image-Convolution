#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include "../../include/image_utils.h"

// ─── Kernel definitions ───────────────────────────────────────────────────────

float edge_detection_3x3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
float sharpen_3x3[9] = {
    0, -1, 0,
   -1,  5, -1,
    0, -1, 0
};

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

// ─── Per-pixel kernel application (same as serial) ───────────────────────────

unsigned char apply_kernel(Image *img, int x, int y, int channel,
                            float *kernel, int kernel_size) {
    float sum = 0.0f;
    int half = kernel_size / 2;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int img_x = x + kx;
            int img_y = y + ky;
            if (img_x < 0)            img_x = 0;
            if (img_x >= img->width)  img_x = img->width  - 1;
            if (img_y < 0)            img_y = 0;
            if (img_y >= img->height) img_y = img->height - 1;
            int img_index    = (img_y * img->width + img_x) * img->channels + channel;
            int kernel_index = (ky + half) * kernel_size + (kx + half);
            sum += img->data[img_index] * kernel[kernel_index];
        }
    }
    if (sum <   0) sum =   0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

// ─── Thread argument struct ───────────────────────────────────────────────────

typedef struct {
    Image        *input;
    Image        *output;
    float        *kernel;
    int           kernel_size;
    int           start_row;
    int           end_row;
} ThreadArgs;

// ─── Thread worker: processes assigned rows ───────────────────────────────────

void* convolve_worker(void *arg) {
    ThreadArgs *args = (ThreadArgs*)arg;
    Image *input      = args->input;
    Image *output     = args->output;
    float *kernel     = args->kernel;
    int kernel_size   = args->kernel_size;

    for (int y = args->start_row; y < args->end_row; y++) {
        for (int x = 0; x < input->width; x++) {
            for (int c = 0; c < input->channels; c++) {
                int index = (y * input->width + x) * input->channels + c;
                output->data[index] = apply_kernel(input, x, y, c, kernel, kernel_size);
            }
        }
    }
    return NULL;
}

// ─── POSIX parallel convolution ──────────────────────────────────────────────

Image* convolve_posix(Image *input, float *kernel, int kernel_size, int num_threads) {
    Image *output = (Image*)malloc(sizeof(Image));
    output->width    = input->width;
    output->height   = input->height;
    output->channels = input->channels;
    output->data     = (unsigned char*)malloc(
        input->width * input->height * input->channels
    );

    pthread_t   *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs  *args    = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));

    int rows_per_thread = input->height / num_threads;
    int remaining_rows  = input->height % num_threads;

    int current_row = 0;
    for (int t = 0; t < num_threads; t++) {
        args[t].input       = input;
        args[t].output      = output;
        args[t].kernel      = kernel;
        args[t].kernel_size = kernel_size;
        args[t].start_row   = current_row;
        args[t].end_row     = current_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
        current_row         = args[t].end_row;

        pthread_create(&threads[t], NULL, convolve_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(args);
    return output;
}
