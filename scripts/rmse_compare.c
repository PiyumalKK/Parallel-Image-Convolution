// rmse_compare.c — Compare two images and compute RMSE + max pixel difference
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/image_utils.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_A> <image_B>\n", argv[0]);
        return 1;
    }

    Image *a = load_image(argv[1]);
    Image *b = load_image(argv[2]);

    if (!a || !b) {
        fprintf(stderr, "ERROR: Failed to load one or both images\n");
        return 1;
    }

    if (a->width != b->width || a->height != b->height || a->channels != b->channels) {
        fprintf(stderr, "ERROR: Image dimensions do not match\n");
        fprintf(stderr, "  A: %dx%d, %d ch\n", a->width, a->height, a->channels);
        fprintf(stderr, "  B: %dx%d, %d ch\n", b->width, b->height, b->channels);
        return 1;
    }

    long total_pixels = (long)a->width * a->height * a->channels;
    double sum_sq = 0.0;
    int max_diff = 0;

    for (long i = 0; i < total_pixels; i++) {
        int diff = (int)a->data[i] - (int)b->data[i];
        if (diff < 0) diff = -diff;
        if (diff > max_diff) max_diff = diff;
        sum_sq += (double)diff * diff;
    }

    double rmse = sqrt(sum_sq / total_pixels);

    printf("RMSE=%.4f MaxDiff=%d Pixels=%ld\n", rmse, max_diff, total_pixels);

    free_image(a);
    free_image(b);
    return 0;
}
