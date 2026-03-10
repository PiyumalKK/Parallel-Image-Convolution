#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include "../../include/image_utils.h"

// Generate normalized Gaussian kernel
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

    for (int i = 0; i < size * size; i++)
        kernel[i] /= sum;

    return kernel;
}

float edge_detection_3x3[9] = {
    -1,-1,-1,
    -1, 8,-1,
    -1,-1,-1
};

float sharpen_3x3[9] = {
     0,-1, 0,
    -1, 5,-1,
     0,-1, 0
};

// Apply convolution kernel to a pixel
unsigned char apply_kernel(Image *img, int x, int y, int c, float *kernel, int ksize) {

    int half = ksize / 2;
    float sum = 0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {

            int img_x = x + kx;
            int img_y = y + ky;

            if (img_x < 0) img_x = 0;
            if (img_x >= img->width) img_x = img->width - 1;
            if (img_y < 0) img_y = 0;
            if (img_y >= img->height) img_y = img->height - 1;

            int img_index = (img_y * img->width + img_x) * img->channels + c;
            int kernel_index = (ky + half) * ksize + (kx + half);

            sum += img->data[img_index] * kernel[kernel_index];
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;

    return (unsigned char)sum;
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s <input_image> <output_image> <filter_type>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Image *input = NULL;
    Image *output = NULL;

    float *kernel = NULL;
    int kernel_size;
    int is_blur = 0;

    int width, height, channels;

    // Root loads image and prepares kernel
    if (rank == 0) {

        input = load_image(argv[1]);
        if (!input) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        width = input->width;
        height = input->height;
        channels = input->channels;

        if (strcmp(argv[3], "blur") == 0) {
            kernel_size = 21;
            kernel = generate_gaussian_kernel(kernel_size, 7.0f);
            is_blur = 1;
        }
        else if (strcmp(argv[3], "edge") == 0) {
            kernel = edge_detection_3x3;
            kernel_size = 3;
        }
        else {
            kernel = sharpen_3x3;
            kernel_size = 3;
        }

        printf("Running MPI with %d processes\n", size);
    }

    // Broadcast image info
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast kernel size
    MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));

    MPI_Bcast(kernel, kernel_size * kernel_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int rows_per_proc = height / size;
    int local_pixels = rows_per_proc * width * channels;

    unsigned char *local_input = (unsigned char*)malloc(local_pixels);
    unsigned char *local_output = (unsigned char*)malloc(local_pixels);

    MPI_Scatter(
        input ? input->data : NULL,
        local_pixels,
        MPI_UNSIGNED_CHAR,
        local_input,
        local_pixels,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    Image local_img;
    local_img.width = width;
    local_img.height = rows_per_proc;
    local_img.channels = channels;
    local_img.data = local_input;

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    // Perform convolution
    for (int y = 0; y < rows_per_proc; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {

                int index = (y * width + x) * channels + c;

                local_output[index] =
                    apply_kernel(&local_img, x, y, c, kernel, kernel_size);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        output = (Image*)malloc(sizeof(Image));
        output->width = width;
        output->height = height;
        output->channels = channels;
        output->data = (unsigned char*)malloc(width * height * channels);
    }

    MPI_Gather(
        local_output,
        local_pixels,
        MPI_UNSIGNED_CHAR,
        output ? output->data : NULL,
        local_pixels,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {

        printf("MPI convolution took: %.4f seconds\n", end - start);

        save_image(argv[2], output);

        free_image(input);
        free_image(output);

        if (is_blur)
            free(kernel);
    }

    free(local_input);
    free(local_output);

    if (rank != 0)
        free(kernel);

    MPI_Finalize();

    return 0;
}