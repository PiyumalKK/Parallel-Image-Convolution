#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Include image utilities as C
extern "C" {
    #include "../include/image_utils.h"
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Maximum kernel size supported (21x21 = 441 elements)
#define MAX_KERNEL_SIZE 21
#define MAX_KERNEL_ELEMS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)

// Constant memory for convolution kernel
// Constant memory is cached and broadcast to all threads in a warp,
// making it ideal for read-only data accessed uniformly by all threads.
__constant__ float d_const_kernel[MAX_KERNEL_ELEMS];

// Thread block dimensions for shared memory tiling
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Generate a true normalized Gaussian kernel (size must be odd)
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
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
    return kernel;
}

float edge_detection_3x3[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
float sharpen_3x3[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};

// CUDA kernel with shared memory tiling
// Each thread block loads a tile of input pixels (including halo region) into
// shared memory, reducing redundant global memory accesses for overlapping
// convolution neighborhoods.
__global__ void convolution_shared(unsigned char *input, unsigned char *output,
                                   int width, int height, int channels,
                                   int kernel_size) {
    int half = kernel_size / 2;

    // Shared memory tile dimensions include halo pixels
    int shared_width = TILE_WIDTH + 2 * half;
    int shared_height = TILE_HEIGHT + 2 * half;

    // Dynamically allocated shared memory for shared_mem_size bytes
    extern __shared__ unsigned char s_tile[];

    // Output pixel coordinates to compute for this thread
    int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int out_y = blockIdx.y * TILE_HEIGHT + threadIdx.y;

    // Load block into shared memory (cooperative loading)
    // Each thread may need to load multiple elements to fill the halo
    for (int c = 0; c < channels; c++) {
        // Number of elements to load per channel
        int tile_elems = shared_width * shared_height;
        int threads_per_block = TILE_WIDTH * TILE_HEIGHT;
        int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;

        // according to the thread id, each thread will load multiple pixels into shared memory to cover the entire tile (including halo)
        for (int idx = tid; idx < tile_elems; idx += threads_per_block) {
            int ty = idx / shared_width;
            int tx = idx % shared_width;

            // Map tile coordinates to image coordinates
            int img_x = blockIdx.x * TILE_WIDTH + tx - half;
            int img_y = blockIdx.y * TILE_HEIGHT + ty - half;

            // Clamp to image boundaries
            img_x = max(0, min(img_x, width - 1));
            img_y = max(0, min(img_y, height - 1));

            // Load pixel into shared memory by all threads in the block cooperatively
            s_tile[c * tile_elems + ty * shared_width + tx] =
                input[(img_y * width + img_x) * channels + c];
        }
    }

    // Synchronize to ensure all threads have loaded their data to shared memory before any thread accesses it
    __syncthreads();

    // Check if this thread's output pixel is within image bounds
    if (out_x >= width || out_y >= height) return;

    // Number of elements in the shared memory tile for one channel
    int tile_elems = shared_width * shared_height;

    // Compute convolution for this output pixel using the shared memory tile and constant memory kernel
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int sx = threadIdx.x + kx;
                int sy = threadIdx.y + ky;

                // Read from shared memory (fast) instead of global memory
                unsigned char pixel = s_tile[c * tile_elems + sy * shared_width + sx];
                // Read kernel from constant memory (cached, broadcast)
                float weight = d_const_kernel[ky * kernel_size + kx];
                sum += pixel * weight;
            }
        }

        // Clamp output to [0, 255]
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        // Write result to global memory
        int out_index = (out_y * width + out_x) * channels + c;
        // Cast to unsigned char and write to output
        output[out_index] = (unsigned char)sum;
    }
}

// Main CUDA convolution function
Image* convolve_cuda(Image *input, float *kern, int kernel_size) {
    //reading the image's properties and calculating total memory needed.
    int width = input->width;
    int height = input->height;
    int channels = input->channels;
    // Total size of the image data in bytes
    size_t img_size = width * height * channels * sizeof(unsigned char);

    // Creating an empty output image (same size as input) on the CPU to store the result later.
    Image *output = (Image*)malloc(sizeof(Image));
    output->width = width;
    output->height = height;
    output->channels = channels;
    output->data = (unsigned char*)malloc(img_size);

    // Copy convolution kernel to constant memory
    // cudaMemcpyToSymbol copies data to __constant__ memory space on device
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, kern,
               kernel_size * kernel_size * sizeof(float)));

    // Allocate device memory for input and output images
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, img_size));
    CUDA_CHECK(cudaMalloc(&d_output, img_size));

    // Copy input image to device (Host-to-Device transfer)
    CUDA_CHECK(cudaMemcpy(d_input, input->data, img_size, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    // Block: TILE_WIDTH x TILE_HEIGHT threads (256 threads per block)
    // Grid: enough blocks to cover the entire image
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH,
              (height + TILE_HEIGHT - 1) / TILE_HEIGHT);

    // Calculate shared memory size for the tiled kernel
    int half = kernel_size / 2;
    int shared_width = TILE_WIDTH + 2 * half;
    int shared_height = TILE_HEIGHT + 2 * half;
    size_t shared_mem_size = shared_width * shared_height * channels * sizeof(unsigned char);

    // Launch kernel with shared memory tiling
    convolution_shared<<<grid, block, shared_mem_size>>>(
        d_input, d_output, width, height, channels, kernel_size);
    //  check whether the GPU accept to run the kernel function and if there is any error during the kernel execution.
    CUDA_CHECK(cudaGetLastError());
    // wait until finish all the threads in the kernel to finish their work before moving on to the next step. This is important to ensure that the output data is ready before we copy it back to the host.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host (Device-to-Host transfer)
    CUDA_CHECK(cudaMemcpy(output->data, d_output, img_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input_image> <output_image> <filter_type>\n", argv[0]);
        printf("filter_type: blur, edge, sharpen\n");
        return 1;
    }

    // Query and print GPU device properties
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA-capable device found\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Constant memory: %zu bytes\n", prop.totalConstMem);

    // Load image
    Image *input = load_image(argv[1]);
    if (!input) return 1;

    printf("Image: %dx%d, %d channels\n", input->width, input->height, input->channels);

    // Select convolution kernel based on filter type
    float *kern;
    int kernel_size;
    int is_blur = 0;

    if (strcmp(argv[3], "blur") == 0) {
        kernel_size = 21;
        float sigma = 7.0f;
        kern = generate_gaussian_kernel(kernel_size, sigma);
        is_blur = 1;
        printf("Filter: Gaussian Blur (kernel=%dx%d, sigma=%.1f)\n", kernel_size, kernel_size, sigma);
    } else if (strcmp(argv[3], "edge") == 0) {
        kern = edge_detection_3x3;
        kernel_size = 3;
        printf("Filter: Edge Detection (kernel=3x3)\n");
    } else if (strcmp(argv[3], "sharpen") == 0) {
        kern = sharpen_3x3;
        kernel_size = 3;
        printf("Filter: Sharpen (kernel=3x3)\n");
    } else {
        fprintf(stderr, "Error: Unknown filter '%s'. Use blur, edge, or sharpen.\n", argv[3]);
        free_image(input);
        return 1;
    }

    // Verify kernel size fits in constant memory
    if (kernel_size > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Error: Kernel size %d exceeds maximum %d\n", kernel_size, MAX_KERNEL_SIZE);
        free_image(input);
        if (is_blur) free(kern);
        return 1;
    }

    // Create CUDA events for timing
    // CUDA events provide GPU-accurate timing without CPU synchronization overhead
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Image *output = convolve_cuda(input, kern, kernel_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("CUDA convolution took: %.4f seconds\n", milliseconds / 1000.0f);

    save_image(argv[2], output);
    printf("Output saved to: %s\n", argv[2]);

    // Cleanup
    free_image(input);
    free_image(output);
    if (is_blur) free(kern);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Reset device
    cudaDeviceReset();

    return 0;
}
