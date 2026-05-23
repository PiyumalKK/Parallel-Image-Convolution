// hybrid_mpi_openmp.c — clamping version (no halo exchange)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "../../include/image_utils.h"

// ─── Kernel generators ────────────────────────────────────────────────────────

float* generate_gaussian_kernel(int size, float sigma) {
    float *kernel = (float*)malloc(size * size * sizeof(float));
    if (!kernel) return NULL;
    int half = size / 2;
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y+half)*size + (x+half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size*size; i++) kernel[i] /= sum;
    return kernel;
}

float edge_kernel[9]    = { -1,-1,-1, -1,8,-1, -1,-1,-1 };
float sharpen_kernel[9] = {  0,-1, 0, -1,5,-1,  0,-1, 0 };

// ─── Single pixel convolution with local clamping ────────────────────────────
/*
 * CLAMPING APPROACH:
 * When the kernel window goes outside the local chunk boundaries,
 * instead of fetching from a neighbour, we clamp the row index
 * to stay within [0, local_rows-1].
 *
 * This means border pixels at chunk seams use repeated edge values
 * rather than the true neighbouring rows. It produces a tiny RMSE
 * at chunk boundaries (same as the pure MPI version) but eliminates
 * all inter-rank communication during computation.
 */
unsigned char apply_kernel_local(
    unsigned char *data, int width, int local_rows, int channels,
    int x, int y, int c,
    float *kernel, int ksize)
{
    float sum = 0.0f;
    int half = ksize / 2;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            // Clamp x to [0, width-1]
            if (ix < 0)          ix = 0;
            if (ix >= width)     ix = width - 1;

            // Clamp y to [0, local_rows-1]
            // At seam boundaries this repeats the edge row of the
            // local chunk instead of reading from the neighbour rank
            if (iy < 0)          iy = 0;
            if (iy >= local_rows) iy = local_rows - 1;

            sum += data[(iy * width + ix) * channels + c]
                   * kernel[(ky+half)*ksize + (kx+half)];
        }
    }
    if (sum <   0) sum =   0;
    if (sum > 255) sum = 255;
    return (unsigned char)sum;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {

    // ═══════════════════════════════════════════════════════════════════
    // STEP 1: Initialise MPI with thread support
    // ═══════════════════════════════════════════════════════════════════
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "ERROR: MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s <input> <output> <blur|edge|sharpen>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 2: Rank 0 loads image; broadcast dimensions
    // ═══════════════════════════════════════════════════════════════════
    int width = 0, height = 0, channels = 0;
    unsigned char *full_image = NULL;

    if (rank == 0) {
        Image *img = load_image(argv[1]);
        if (!img) {
            fprintf(stderr, "ERROR: Failed to load image: %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        width      = img->width;
        height     = img->height;
        channels   = img->channels;
        full_image = img->data;
        img->data  = NULL;
        free(img);
    }

    MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ═══════════════════════════════════════════════════════════════════
    // STEP 3: Build kernel on rank 0, broadcast to all ranks
    // ═══════════════════════════════════════════════════════════════════
    float *kernel = NULL;
    int    ksize  = 0;
    int    is_blur = 0;

    if (rank == 0) {
        if (strcmp(argv[3], "blur") == 0) {
            ksize  = 21;
            kernel = generate_gaussian_kernel(ksize, 7.0f);
            if (!kernel) {
                fprintf(stderr, "ERROR: Failed to generate Gaussian kernel\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            is_blur = 1;
        } else if (strcmp(argv[3], "edge") == 0) {
            kernel = edge_kernel;
            ksize  = 3;
        } else {
            kernel = sharpen_kernel;
            ksize  = 3;
        }
    }

    MPI_Bcast(&ksize,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&is_blur, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        kernel = (float*)malloc(ksize * ksize * sizeof(float));

    MPI_Bcast(kernel, ksize * ksize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // ═══════════════════════════════════════════════════════════════════
    // STEP 4: Start total wall-clock timer
    // ═══════════════════════════════════════════════════════════════════
    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 5: Divide image rows across MPI ranks
    // ═══════════════════════════════════════════════════════════════════
    int *row_counts  = (int*)malloc(size * sizeof(int));
    int *row_offsets = (int*)malloc(size * sizeof(int));

    int base_rows = height / size;
    int remainder = height % size;

    for (int i = 0; i < size; i++) {
        row_counts[i]  = base_rows + (i < remainder ? 1 : 0);
        row_offsets[i] = (i == 0) ? 0 : row_offsets[i-1] + row_counts[i-1];
    }

    int local_rows = row_counts[rank];

    // ═══════════════════════════════════════════════════════════════════
    // STEP 6: Scatter image rows to all ranks
    // ═══════════════════════════════════════════════════════════════════
    /*
     * With clamping there are NO halo rows to exchange.
     * Each rank only receives exactly its own local_rows — nothing extra.
     * This is simpler and requires less memory per rank.
     */
    int *send_counts = (int*)malloc(size * sizeof(int));
    int *send_displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        send_counts[i] = row_counts[i]  * width * channels;
        send_displs[i] = row_offsets[i] * width * channels;
    }

    unsigned char *local_chunk = (unsigned char*)malloc(
        local_rows * width * channels);
    if (!local_chunk) {
        fprintf(stderr, "ERROR: Rank %d failed to allocate local_chunk\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(
        full_image, send_counts, send_displs, MPI_UNSIGNED_CHAR,
        local_chunk, local_rows * width * channels, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // ═══════════════════════════════════════════════════════════════════
    // STEP 7: OpenMP parallel convolution with clamping
    // ═══════════════════════════════════════════════════════════════════
    /*
     * No halo exchange needed before this step.
     * apply_kernel_local handles out-of-bounds by clamping to the
     * local chunk edges. Pixels at the seam between two ranks will
     * use the repeated edge row value rather than the true neighbour —
     * this introduces a tiny RMSE at boundaries but is much simpler.
     *
     * Thread count read from OMP_NUM_THREADS environment variable.
     * Set this before running:
     *   export OMP_NUM_THREADS=4   (Linux)
     *   $env:OMP_NUM_THREADS = 4   (Windows PowerShell)
     */
    int omp_threads = omp_get_max_threads();

    unsigned char *local_output = (unsigned char*)malloc(
        local_rows * width * channels);
    if (!local_output) {
        fprintf(stderr, "ERROR: Rank %d failed to allocate local_output\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double omp_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic) num_threads(omp_threads)
    for (int y = 0; y < local_rows; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int out_idx = (y * width + x) * channels + c;
                local_output[out_idx] = apply_kernel_local(
                    local_chunk,
                    width, local_rows, channels,
                    x, y, c,           // plain y — no halo offset needed
                    kernel, ksize
                );
            }
        }
    }

    double omp_end = omp_get_wtime();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 8: Gather all processed chunks back to rank 0
    // ═══════════════════════════════════════════════════════════════════
    unsigned char *output_image = NULL;
    if (rank == 0)
        output_image = (unsigned char*)malloc(height * width * channels);

    MPI_Gatherv(
        local_output, local_rows * width * channels, MPI_UNSIGNED_CHAR,
        output_image, send_counts, send_displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double total_end = MPI_Wtime();

    // ═══════════════════════════════════════════════════════════════════
    // STEP 9: Collect timing across all ranks
    // ═══════════════════════════════════════════════════════════════════
    double local_omp_time = omp_end - omp_start;
    double max_omp_time   = 0.0;
    MPI_Reduce(&local_omp_time, &max_omp_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ═══════════════════════════════════════════════════════════════════
    // STEP 10: Rank 0 saves image and prints timing
    // ═══════════════════════════════════════════════════════════════════
    if (rank == 0) {
        Image out_img;
        out_img.width    = width;
        out_img.height   = height;
        out_img.channels = channels;
        out_img.data     = output_image;
        save_image(argv[2], &out_img);

        printf("Hybrid MPI+OpenMP convolution took: %.4f seconds\n",
               total_end - total_start);
        printf("  Filter          : %s\n",    argv[3]);
        printf("  Kernel size     : %dx%d\n", ksize, ksize);
        printf("  MPI ranks       : %d\n",    size);
        printf("  OMP threads/rank: %d\n",    omp_threads);
        printf("  Total threads   : %d\n",    size * omp_threads);
        printf("  Max OMP compute : %.4f seconds\n", max_omp_time);
        printf("  MPI overhead    : %.4f seconds\n",
               (total_end - total_start) - max_omp_time);
        printf("  Note: clamping used at chunk boundaries (small RMSE expected)\n");

        free(output_image);
        free(full_image);
    }

    // ═══════════════════════════════════════════════════════════════════
    // STEP 11: Clean up
    // ═══════════════════════════════════════════════════════════════════
    free(local_chunk);
    free(local_output);
    free(row_counts);
    free(row_offsets);
    free(send_counts);
    free(send_displs);
    if (is_blur) free(kernel);

    MPI_Finalize();
    return 0;
}