#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include "../include/image_utils.h"

// Recursively create directories for the given file path
static void ensure_parent_dirs(const char *filepath) {
    char *path = strdup(filepath);
    for (char *p = path + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            mkdir(path, 0755);
            *p = '/';
        }
    }
    free(path);
}

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
    ensure_parent_dirs(filename);
    stbi_write_png(filename, img->width, img->height, img->channels, img->data, img->width * img->channels);
}

void free_image(Image *img) {
    if (img) {
        if (img->data) stbi_image_free(img->data);
        free(img);
    }
}
