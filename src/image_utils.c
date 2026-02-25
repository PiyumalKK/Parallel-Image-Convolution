#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_utils.h"

Image* load_image(const char *filename) {
    Image *img = (Image*)malloc(sizeof(Image));
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 3);
    if (!img->data) {
        printf("Error loading image: %s\n", filename);
        free(img);
        return NULL;
    }
    img->channels = 3; // Force RGB
    return img;
}

void save_image(const char *filename, Image *img) {
    stbi_write_png(filename, img->width, img->height, img->channels, img->data, img->width * img->channels);
}

void free_image(Image *img) {
    if (img) {
        if (img->data) stbi_image_free(img->data);
        free(img);
    }
}
