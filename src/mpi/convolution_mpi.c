#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../../include/image_utils.h"

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

    int width, height, channels;

    if (rank == 0) {

        input = load_image(argv[1]);
        if (!input) {
            printf("Error loading image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        width = input->width;
        height = input->height;
        channels = input->channels;

        printf("Running MPI with %d processes\n", size);
    }

    MPI_Finalize();
    return 0;
}