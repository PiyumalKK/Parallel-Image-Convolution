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

/*
 * Generate a normalized Gaussian kernel.
 * Rule of thumb: kernel_size should be ceil(6*sigma) | 1  (odd, at least 6σ wide)
 * to avoid truncating meaningful Gaussian weights.
 */
float* generate_gaussian_kernel(int size, float sigma) {
    float *kernel = (float*)malloc(size * size * sizeof(float));
    if (!kernel) return NULL;
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

// ─── Grayscale conversion helpers ────────────────────────────────────────────

/*
 * Convert an RGB image to single-channel grayscale (luminance).
 * Uses ITU-R BT.601 coefficients: Y = 0.299R + 0.587G + 0.114B
 * Returns a newly allocated single-channel Image, or NULL on failure.
 * Caller must free_image() the result.
 */
Image* rgb_to_grayscale(Image *img) {
    if (img->channels == 1) {
        /* Already grayscale — return a deep copy */
        Image *copy = (Image*)malloc(sizeof(Image));
        if (!copy) return NULL;
        copy->width    = img->width;
        copy->height   = img->height;
        copy->channels = 1;
        size_t sz      = (size_t)img->width * img->height;
        copy->data     = (unsigned char*)malloc(sz);
        if (!copy->data) { free(copy); return NULL; }
        memcpy(copy->data, img->data, sz);
        return copy;
    }

    Image *gray = (Image*)malloc(sizeof(Image));
    if (!gray) return NULL;
    gray->width    = img->width;
    gray->height   = img->height;
    gray->channels = 1;
    size_t sz      = (size_t)img->width * img->height;
    gray->data     = (unsigned char*)malloc(sz);
    if (!gray->data) { free(gray); return NULL; }

    for (int i = 0; i < img->width * img->height; i++) {
        float r = img->data[i * img->channels + 0];
        float g = img->data[i * img->channels + 1];
        float b = img->data[i * img->channels + 2];
        gray->data[i] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b);
    }
    return gray;
}

/*
 * Copy a single-channel grayscale image back into a 3-channel RGB image
 * by replicating the luminance into R, G, and B.
 * Returns a new Image, or NULL on failure.
 */
Image* grayscale_to_rgb(Image *gray, int target_channels) {
    if (target_channels == 1) {
        return rgb_to_grayscale(gray);   /* trivial copy via existing helper */
    }
    Image *rgb = (Image*)malloc(sizeof(Image));
    if (!rgb) return NULL;
    rgb->width    = gray->width;
    rgb->height   = gray->height;
    rgb->channels = target_channels;
    size_t sz     = (size_t)gray->width * gray->height * target_channels;
    rgb->data     = (unsigned char*)malloc(sz);
    if (!rgb->data) { free(rgb); return NULL; }

    for (int i = 0; i < gray->width * gray->height; i++) {
        for (int c = 0; c < target_channels; c++)
            rgb->data[i * target_channels + c] = gray->data[i];
    }
    return rgb;
}

// ─── Per-pixel kernel application ────────────────────────────────────────────

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
    if (num_threads < 1) num_threads = 1;
    if (num_threads > input->height) num_threads = input->height;

    Image *output = (Image*)malloc(sizeof(Image));
    if (!output) {
        fprintf(stderr, "ERROR: Failed to allocate output image struct\n");
        return NULL;
    }

    output->width    = input->width;
    output->height   = input->height;
    output->channels = input->channels;
    size_t data_size = (size_t)input->width * input->height * input->channels;
    output->data     = (unsigned char*)malloc(data_size);

    if (!output->data) {
        fprintf(stderr, "ERROR: Failed to allocate output image data\n");
        free(output);
        return NULL;
    }

    pthread_t   *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadArgs  *args    = (ThreadArgs*)malloc(num_threads * sizeof(ThreadArgs));

    if (!threads || !args) {
        fprintf(stderr, "ERROR: Failed to allocate thread structures\n");
        free(output->data);
        free(output);
        free(threads);
        free(args);
        return NULL;
    }

    // Initialize thread attribute and explicitly set joinable
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    int rows_per_thread = input->height / num_threads;
    int remaining_rows  = input->height % num_threads;
    int current_row     = 0;

    /* Initialise ALL thread arguments before spawning any thread */
    for (int t = 0; t < num_threads; t++) {
        args[t].input       = input;
        args[t].output      = output;
        args[t].kernel      = kernel;
        args[t].kernel_size = kernel_size;
        args[t].start_row   = current_row;
        args[t].end_row     = current_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
        current_row         = args[t].end_row;
    }

    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_create(&threads[t], &attr, convolve_worker, &args[t]);
        if (rc) {
            fprintf(stderr, "ERROR: pthread_create() returned %d\n", rc);
            pthread_attr_destroy(&attr);
            free(threads);
            free(args);
            free(output->data);
            free(output);
            exit(-1);
        }
    }

    pthread_attr_destroy(&attr);
    for (int t = 0; t < num_threads; t++) {
        int rc = pthread_join(threads[t], NULL);
        if (rc) {
            fprintf(stderr, "ERROR: pthread_join() returned %d\n", rc);
            exit(-1);
        }
    }

    free(threads);
    free(args);
    return output;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <input_image> <output_image> <filter_type> <num_threads>\n", argv[0]);
        printf("filter_type: blur, edge, sharpen\n");
        return 1;
    }

    Image *input = load_image(argv[1]);
    if (!input) {
        fprintf(stderr, "ERROR: Failed to load input image\n");
        return 1;
    }

    /* Robust thread count parsing */
    char *endptr;
    long  num_threads_l = strtol(argv[4], &endptr, 10);
    if (*endptr != '\0' || num_threads_l < 1) {
        fprintf(stderr, "WARNING: Invalid thread count '%s', defaulting to 1\n", argv[4]);
        num_threads_l = 1;
    }
    int num_threads = (int)num_threads_l;

    float *kernel;
    int    kernel_size;
    int    is_blur      = 0;
    int    is_edge      = 0;

    if (strcmp(argv[3], "blur") == 0) {
        float sigma = 7.0f;
        kernel_size = 21;      // consistent with serial/openmp/mpi/cuda
        kernel      = generate_gaussian_kernel(kernel_size, sigma);
        if (!kernel) {
            fprintf(stderr, "ERROR: Failed to generate Gaussian kernel\n");
            free_image(input);
            return 1;
        }
        is_blur = 1;

    } else if (strcmp(argv[3], "edge") == 0) {
        kernel      = edge_detection_3x3;
        kernel_size = 3;
        is_edge     = 1;

    } else {
        /* sharpen */
        kernel      = sharpen_3x3;
        kernel_size = 3;
    }

    struct timespec start_time, end_time;
    Image *output = NULL;

    if (is_edge) {
        /*
         * FIX: Edge detection is a Laplacian operator that measures intensity
         * gradients. It should be applied to a single luminance channel, not
         * independently per colour channel, to avoid colour fringing artifacts.
         *
         * Workflow:
         *   1. Convert to grayscale
         *   2. Convolve the single-channel image
         *   3. Expand back to the original channel count for saving
         */
        Image *gray = rgb_to_grayscale(input);
        if (!gray) {
            fprintf(stderr, "ERROR: Grayscale conversion failed\n");
            free_image(input);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &start_time);
        Image *edge_gray = convolve_posix(gray, kernel, kernel_size, num_threads);
        clock_gettime(CLOCK_MONOTONIC, &end_time);

        free_image(gray);

        if (!edge_gray) {
            fprintf(stderr, "ERROR: Edge convolution failed\n");
            free_image(input);
            return 1;
        }

        /* Replicate grayscale result into RGB so save_image works correctly */
        output = grayscale_to_rgb(edge_gray, input->channels);
        free_image(edge_gray);

        if (!output) {
            fprintf(stderr, "ERROR: Failed to expand edge result to RGB\n");
            free_image(input);
            return 1;
        }

    } else {
        /* Blur and sharpen operate on all channels directly — no conversion needed */
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        output = convolve_posix(input, kernel, kernel_size, num_threads);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
    }

    if (!output) {
        fprintf(stderr, "ERROR: Convolution failed\n");
        free_image(input);
        if (is_blur) free(kernel);
        return 1;
    }

    double time_taken = (end_time.tv_sec  - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("POSIX convolution (%d threads, filter=%s, kernel=%dx%d) took: %.4f seconds\n",
           num_threads, argv[3], kernel_size, kernel_size, time_taken);

    save_image(argv[2], output);

    free_image(input);
    free_image(output);
    if (is_blur) free(kernel);

    return 0;
}