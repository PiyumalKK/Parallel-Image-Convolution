#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>                          // ← added
#include "../../include/image_utils.h"

/* ── Kernels ── */

float* generate_gaussian_kernel(int size, float sigma) {
    float *kernel = malloc(size * size * sizeof(float));
    int half = size / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++)
        for (int x = -half; x <= half; x++) {
            float v = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y+half)*size + (x+half)] = v;
            sum += v;
        }
    for (int i = 0; i < size*size; i++) kernel[i] /= sum;
    return kernel;
}

float edge_detection_3x3[9] = { -1,-1,-1, -1,8,-1, -1,-1,-1 };
float sharpen_3x3[9]        = {  0,-1, 0, -1,5,-1,  0,-1, 0 };

/* ── Apply kernel to one pixel (thread-safe: read-only on all_data) ── */

static inline unsigned char apply_kernel(
    const unsigned char *data,
    int width, int height, int channels,
    int x, int y, int channel,
    const float *kernel, int ksz)
{
    float sum = 0.0f;
    int half = ksz / 2;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx; if (ix < 0) ix = 0; if (ix >= width)  ix = width-1;
            int iy = y + ky; if (iy < 0) iy = 0; if (iy >= height) iy = height-1;
            sum += data[(iy*width + ix)*channels + channel]
                 * kernel[(ky+half)*ksz + (kx+half)];
        }
    }
    return (unsigned char)(sum < 0 ? 0 : sum > 255 ? 255 : sum);
}

/* ── Main ── */

int main(int argc, char *argv[]) {

    // MPI_THREAD_FUNNELED: only main thread calls MPI
    // OpenMP threads run freely inside each rank      ← changed from MPI_Init
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s <input> <output> <blur|edge|sharpen>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* ── Step 1: Rank 0 loads image ── */
    int width = 0, height = 0, channels = 0;
    unsigned char *full_data = NULL;
    float *kernel = NULL;
    int kernel_size = 0, is_blur = 0;

    if (rank == 0) {
        Image *img = load_image(argv[1]);
        if (!img) { MPI_Abort(MPI_COMM_WORLD, 1); }
        width     = img->width;
        height    = img->height;
        channels  = img->channels;
        full_data = img->data;
        free(img);

        if (strcmp(argv[3], "blur") == 0) {
            kernel_size = 21;
            kernel = generate_gaussian_kernel(kernel_size, 7.0f);
            is_blur = 1;
        } else if (strcmp(argv[3], "edge") == 0) {
            kernel = edge_detection_3x3; kernel_size = 3;
        } else {
            kernel = sharpen_3x3; kernel_size = 3;
        }
    }

    /* ── Step 2: Broadcast image info and kernel ── */
    MPI_Bcast(&width,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&is_blur,     1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        kernel = malloc(kernel_size * kernel_size * sizeof(float));

    MPI_Bcast(kernel, kernel_size * kernel_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ── Step 3: Broadcast full image to all ranks ── */
    int total_pixels = width * height * channels;
    unsigned char *all_data = malloc(total_pixels);

    if (rank == 0) memcpy(all_data, full_data, total_pixels);
    MPI_Bcast(all_data, total_pixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    /* ── Step 4: Calculate each rank's row range ── */
    int rows_per_rank = height / nprocs;
    int remainder     = height % nprocs;
    int start_row = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    int my_rows   = rows_per_rank + (rank < remainder ? 1 : 0);
    int my_pixels = my_rows * width * channels;

    unsigned char *local_out = malloc(my_pixels);

    /* ── Step 5: OpenMP parallelises pixel loop inside each MPI rank ── */
    //            Each rank spawns OMP_NUM_THREADS threads here
    double t_start = MPI_Wtime();

    #pragma omp parallel for schedule(dynamic, 4) collapse(2)  // ← added
    for (int y = start_row; y < start_row + my_rows; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int local_idx = ((y - start_row) * width + x) * channels + c;
                local_out[local_idx] = apply_kernel(
                    all_data, width, height, channels,
                    x, y, c, kernel, kernel_size);
            }
        }
    }

    double t_end  = MPI_Wtime();
    double local_time = t_end - t_start;

    /* ── Step 6: Gather results to rank 0 ── */
    int *recvcounts = NULL, *displs = NULL;
    unsigned char *gathered = NULL;

    if (rank == 0) {
        recvcounts = malloc(nprocs * sizeof(int));
        displs     = malloc(nprocs * sizeof(int));
        gathered   = malloc(total_pixels);

        for (int r = 0; r < nprocs; r++) {
            int r_start   = r * rows_per_rank + (r < remainder ? r : remainder);
            int r_rows    = rows_per_rank + (r < remainder ? 1 : 0);
            recvcounts[r] = r_rows * width * channels;
            displs[r]     = r_start * width * channels;
        }
    }

    MPI_Gatherv(local_out, my_pixels, MPI_UNSIGNED_CHAR,
                gathered, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    /* ── Step 7: Report timing and save ── */
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Hybrid MPI+OpenMP convolution took : %.4f seconds\n", max_time);
        printf("  MPI ranks                        : %d\n", nprocs);
        printf("  OpenMP threads per rank          : %d\n", omp_get_max_threads()); // ← added
        printf("  Total parallel workers           : %d\n", nprocs * omp_get_max_threads()); // ← added

        Image out;
        out.width    = width;
        out.height   = height;
        out.channels = channels;
        out.data     = gathered;
        save_image(argv[2], &out);

        free(gathered); free(recvcounts); free(displs);
        free(full_data);
        if (is_blur) free(kernel);
    } else {
        if (is_blur) free(kernel);
    }

    free(all_data);
    free(local_out);
    MPI_Finalize();
    return 0;
}