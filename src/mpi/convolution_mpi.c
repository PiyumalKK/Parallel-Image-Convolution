#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../../include/image_utils.h"

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