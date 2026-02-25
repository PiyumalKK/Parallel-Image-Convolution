#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H
#include <stdio.h>
#include <stdlib.h>
typedef struct {
    unsigned char *data;  // RGB pixel data
    int width;
    int height;
    int channels;         // 3 for RGB
} Image;
// Function declarations
Image* load_image(const char *filename);
void save_image(const char *filename, Image *img);
void free_image(Image *img);
#endif
