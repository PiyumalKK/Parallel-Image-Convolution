// hybrid_mpi_openmp.c  — corrected version
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include "../include/image_utils.h"

#define NUM_THREADS 4   // OpenMP threads per MPI rank

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

/*
 * FIX #2: Helper to compute correct kernel size from sigma using the 6σ rule.
 * For σ=2 this gives 13; ensures Gaussian tails are not truncated.
 */
int gaussian_kernel_size(float sigma) {
    int size = (int)ceilf(6.0f * sigma) + 1;
    if (size % 2 == 0) size++;
    if (size < 3) size = 3;
    return size;
}

float edge_kernel[9]    = { -1,-1,-1, -1,8,-1, -1,-1,-1 };
float sharpen_kernel[9] = {  0,-1, 0, -1,5,-1,  0,-1, 0 };

// ─── Single pixel convolution (thread-safe, reads only) ──────────────────────

unsigned char apply_kernel_local(
    unsigned char *data, int width, int height, int channels,
    int x, int y, int c,
    float *kernel, int ksize)
{
    float sum = 0.0f;
    int half = ksize / 2;
    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            if (ix < 0)       ix = 0;
            if (ix >= width)  ix = width  - 1;
            if (iy < 0)       iy = 0;
            if (iy >= height) iy = height - 1;
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

    // ── 1. Init MPI with thread support ──────────────────────────────────────
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

    // ── 2. Rank 0 loads image and broadcasts metadata ─────────────────────────
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

        // FIX #1: Detach data before freeing struct to avoid dangling pointer
        full_image = img->data;
        img->data  = NULL;
        free(img);
    }

    // Broadcast image dimensions to all ranks
    MPI_Bcast(&width,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ── 3. Build kernel on all ranks ─────────────────────────────────────────
    float *kernel = NULL;
    int    ksize  = 0;
    int    is_blur = 0;

    if (strcmp(argv[3], "blur") == 0) {
        // FIX #2: Use sigma=2 and derive correct kernel size from 6σ rule
        float sigma = 2.0f;
        ksize  = gaussian_kernel_size(sigma);   // = 13 for sigma=2
        kernel = generate_gaussian_kernel(ksize, sigma);
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

    // ── 4. Start total wall-clock timer (FIX #5: use MPI_Wtime for full pipeline)
    double total_start = MPI_Wtime();

    // ── 5. Calculate row distribution across ranks ────────────────────────────
    int *row_counts  = (int*)malloc(size * sizeof(int));
    int *row_offsets = (int*)malloc(size * sizeof(int));

    int base_rows = height / size;
    int remainder = height % size;

    for (int i = 0; i < size; i++) {
        row_counts[i]  = base_rows + (i < remainder ? 1 : 0);
        row_offsets[i] = (i == 0) ? 0 : row_offsets[i-1] + row_counts[i-1];
    }

    int local_rows = row_counts[rank];

    // ── 6. Calculate halo (ghost) rows ────────────────────────────────────────
    int half_k      = ksize / 2;
    int top_halo    = (rank > 0)        ? half_k : 0;
    int bottom_halo = (rank < size - 1) ? half_k : 0;
    int halo_rows   = local_rows + top_halo + bottom_halo;

    // ── 7. Scatter image rows to all ranks ────────────────────────────────────
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

    // ── 8. Exchange halo rows between neighbouring ranks ─────────────────────
    unsigned char *halo_buffer = (unsigned char*)malloc(
        halo_rows * width * channels);
    if (!halo_buffer) {
        fprintf(stderr, "ERROR: Rank %d failed to allocate halo_buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Copy own rows into middle of halo_buffer
    memcpy(halo_buffer + top_halo * width * channels,
           local_chunk, local_rows * width * channels);

    // FIX #3: Use consistent matching tags for halo exchange
    // Tag 10 = top halo exchange, Tag 20 = bottom halo exchange

    // Send our top rows to rank-1; receive rank-1's bottom rows as our top halo
    if (rank > 0) {
        MPI_Sendrecv(
            local_chunk,                              // send our top rows up
            half_k * width * channels, MPI_UNSIGNED_CHAR, rank-1, 10,
            halo_buffer,                              // receive top halo
            half_k * width * channels, MPI_UNSIGNED_CHAR, rank-1, 20,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Send our bottom rows to rank+1; receive rank+1's top rows as our bottom halo
    if (rank < size - 1) {
        unsigned char *my_bottom   = local_chunk +
            (local_rows - half_k) * width * channels;
        unsigned char *recv_bottom = halo_buffer +
            (top_halo + local_rows) * width * channels;

        MPI_Sendrecv(
            my_bottom,                                // send our bottom rows down
            half_k * width * channels, MPI_UNSIGNED_CHAR, rank+1, 20,
            recv_bottom,                              // receive bottom halo
            half_k * width * channels, MPI_UNSIGNED_CHAR, rank+1, 10,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // ── 9. OpenMP parallel convolution on local chunk ─────────────────────────
    unsigned char *local_output = (unsigned char*)malloc(
        local_rows * width * channels);
    if (!local_output) {
        fprintf(stderr, "ERROR: Rank %d failed to allocate local_output\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double omp_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
    for (int y = 0; y < local_rows; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int out_idx = (y * width + x) * channels + c;
                local_output[out_idx] = apply_kernel_local(
                    halo_buffer,
                    width, halo_rows, channels,
                    x, y + top_halo, c,
                    kernel, ksize
                );
            }
        }
    }

    double omp_end = omp_get_wtime();

    // ── 10. Gather results back to rank 0 ─────────────────────────────────────
    unsigned char *output_image = NULL;
    if (rank == 0)
        output_image = (unsigned char*)malloc(height * width * channels);

    MPI_Gatherv(
        local_output, local_rows * width * channels, MPI_UNSIGNED_CHAR,
        output_image, send_counts, send_displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    double total_end = MPI_Wtime();

    // ── 11. Rank 0 saves image and reports timing ──────────────────────────────
    if (rank == 0) {
        Image out_img;
        out_img.width    = width;
        out_img.height   = height;
        out_img.channels = channels;
        out_img.data     = output_image;

        // FIX #6: save_image returns void; just call it directly
        save_image(argv[2], &out_img);

        free(output_image);
        free(full_image);

        printf("Hybrid MPI+OpenMP convolution complete.\n");
        printf("  Filter       : %s\n",   argv[3]);
        printf("  Kernel size  : %dx%d\n", ksize, ksize);
        printf("  MPI ranks    : %d\n",   size);
        printf("  OMP threads  : %d per rank\n", NUM_THREADS);
        printf("  Total threads: %d\n",   size * NUM_THREADS);
        printf("  Total wall time: %.4f seconds\n", total_end - total_start);
    }

    // Each rank prints its own OpenMP compute time
    printf("  Rank %d OMP compute time: %.4f seconds\n",
           rank, omp_end - omp_start);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free(local_chunk);
    free(halo_buffer);
    free(local_output);
    free(row_counts);
    free(row_offsets);
    free(send_counts);
    free(send_displs);
    if (is_blur) free(kernel);

    MPI_Finalize();
    return 0;
}