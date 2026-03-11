# Parallel Image Convolution

Image convolution implemented using multiple parallelization approaches — **Serial**, **POSIX Threads**, **OpenMP**, **MPI**, and **CUDA** — to compare performance across CPU and GPU. Each version applies a convolution kernel (filter) to an input image pixel-by-pixel and writes the result to an output image.

## Project Structure

```
project/
│
├── src/
│   ├── image_utils.c              # Image load/save using stb_image
│   |
│   ├── serial/
│   │   └── convolution_serial.c   # Single-threaded CPU implementation
│   |
│   ├── posix/
│   │   └── convolution_posix.c    # Multi-threaded CPU with POSIX threads
│   |
│   ├── openmp/
│   │   └── convolution_openmp.c   # Multi-threaded CPU with OpenMP
│   |
│   ├── mpi/
│   │   └── convolution_mpi.c      # Distributed-memory parallel with MPI
│   |
│   └── cuda/
│       └── convolution_cuda.cu    # GPU implementation using CUDA
│
├── include/
│   ├── image_utils.h              # Image struct and function declarations
│   ├── stb_image.h                # stb image loading library
│   └── stb_image_write.h          # stb image writing library
│
└── images/
    ├── input/                     # Place input images here
    │   ├── test.jpg               # Input image for blur
    │   ├── test_edge.jpg          # Input image for edge detection
    │   └── test_sharp.jpg         # Input image for sharpening
    │
    └── output/                    # Processed images saved here
```

## Supported Filters

| Filter     | Description                                      | Kernel Size |
|------------|--------------------------------------------------|-------------|
| `blur`     | Gaussian blur (sigma=7.0)                        | 21×21       |
| `edge`     | Edge detection (Laplacian)                        | 3×3         |
| `sharpen`  | Sharpening filter                                 | 3×3         |

## Prerequisites

- **GCC** (for Serial, POSIX, and OpenMP builds)
- **NVIDIA CUDA Toolkit** (for CUDA build)
- **MSVC (`cl.exe`)** — required by `nvcc` on Windows. Comes with Visual Studio (Build Tools or Community edition).

## Compilation & Running

All commands assume you are in the project root directory:

```
cd C:\HPC\Parallel-Image-Convolution
```

---

### 1. Serial (Single-threaded CPU)

**Compile:**
```bash
gcc -o convolution_serial src/serial/convolution_serial.c src/image_utils.c -I include -lm
```

**Run:**
```bash
# Blur
./convolution_serial images/input/test.jpg images/output/blur_serial.jpg blur

# Edge Detection
./convolution_serial images/input/test_edge.jpg images/output/edge_serial.jpg edge

# Sharpen
./convolution_serial images/input/test_sharp.jpg images/output/sharp_serial.jpg sharpen
```

---

### 2. OpenMP (Multi-threaded CPU)

**Compile:**
```bash
gcc -fopenmp -o convolution_openmp src/openmp/convolution_openmp.c src/image_utils.c -I include -lm
```

**Run:**
```bash
# Blur
./convolution_openmp images/input/test.jpg images/output/blur_openmp.jpg blur

# Edge Detection
./convolution_openmp images/input/test_edge.jpg images/output/edge_openmp.jpg edge

# Sharpen
./convolution_openmp images/input/test_sharp.jpg images/output/sharp_openmp.jpg sharpen
```

You can control the number of threads with the `OMP_NUM_THREADS` environment variable:
```bash
# Linux/macOS
export OMP_NUM_THREADS=8

# Windows PowerShell
$env:OMP_NUM_THREADS = 8
```

---

### 2. MPI (Distributed-memory CPU)

**Compile:**
```bash
mpicc -o convolution_mpi src/mpi/convolution_mpi.c src/image_utils.c -I include -lm
```

**Run:**
```bash
# Blur
mpirun -np 4 ./convolution_mpi images/input/test.jpg images/output/blur_mpi.jpg blur

# Edge Detection
mpirun -np 4 ./convolution_mpi images/input/test_edge.jpg images/output/edge_mpi.jpg edge

# Sharpen
mpirun -np 4 ./convolution_mpi images/input/test_sharp.jpg images/output/sharp_mpi.jpg sharpen
```

You can control the number of process using the -np option:

```bash
# Linux/macOS
# Run with 4 processes
mpirun -np 4 ./convolution_mpi images/input/test.jpg images/output/blur_mpi.jpg blur

# Run with 8 processes
mpirun -np 8 ./convolution_mpi images/input/test.jpg images/output/blur_mpi.jpg blur
```

---

### 3. POSIX Threads (Multi-threaded CPU)

No extra installation needed — pthreads is built into GCC on Linux/WSL.

**Compile:**
```bash
gcc -o convolution_posix src/posix/convolution_posix.c src/image_utils.c -I include -lm -lpthread
```

**Run (thread count is the 4th argument):**
```bash
# Blur
./convolution_posix images/input/test.jpg images/output/blur_posix.jpg blur 4

# Edge Detection
./convolution_posix images/input/test_edge.jpg images/output/edge_posix.jpg edge 4

# Sharpen
./convolution_posix images/input/test_sharp.jpg images/output/sharp_posix.jpg sharpen 4
```

You can change the thread count by changing the last argument:
```bash
# Run with 8 threads
./convolution_posix images/input/test.jpg images/output/blur_posix_8T.jpg blur 8
```

> **Note:** POSIX version uses `clock()` which measures total **CPU time** across all threads, not wall-clock time. So reported times appear similar to serial even though the actual elapsed time is faster.

---

### 4. CUDA (GPU)

#### Windows Setup

`nvcc` requires the MSVC compiler (`cl.exe`). Add the MSVC `cl.exe` directory to your system **PATH**:

1. Find your `cl.exe` path, typically:
   ```
   C:\Program Files\Microsoft Visual Studio\<version>\<edition>\VC\Tools\MSVC\<toolset>\bin\HostX64\x64
   ```
   For example: `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\HostX64\x64`

2. Open **Settings → System → About → Advanced system settings → Environment Variables**
3. Under **System variables**, edit **Path** and add the directory from step 1
4. Restart VS Code / your terminal

After this, `nvcc` will find `cl.exe` automatically in any terminal.

**Compile:**
```bash
nvcc -allow-unsupported-compiler -o convolution_cuda src/cuda/convolution_cuda.cu src/image_utils.c -I include
```

> The `-allow-unsupported-compiler` flag is needed if your Visual Studio version is newer than what the CUDA Toolkit officially supports. It can be omitted if you use a supported VS version (2019–2022).

**Run:**
```bash
# Blur
./convolution_cuda images/input/test.jpg images/output/blur_cuda.jpg blur

# Edge Detection
./convolution_cuda images/input/test_edge.jpg images/output/edge_cuda.jpg edge

# Sharpen
./convolution_cuda images/input/test_sharp.jpg images/output/sharp_cuda.jpg sharpen
```

---

### Usage Summary

```
<executable> <input_image> <output_image> <filter_type>
```

| Parameter      | Description                                          |
|----------------|------------------------------------------------------|
| `input_image`  | Path to the input image (JPG, PNG, BMP, etc.)        |
| `output_image` | Path to save the filtered output image               |
| `filter_type`  | One of: `blur`, `edge`, `sharpen`                    |

**Examples:**
```bash
# Serial — all filters
./convolution_serial images/input/test.jpg images/output/blur_serial.jpg blur
./convolution_serial images/input/test_edge.jpg images/output/edge_serial.jpg edge
./convolution_serial images/input/test_sharp.jpg images/output/sharp_serial.jpg sharpen

# OpenMP — all filters
./convolution_openmp images/input/test.jpg images/output/blur_openmp.jpg blur
./convolution_openmp images/input/test_edge.jpg images/output/edge_openmp.jpg edge
./convolution_openmp images/input/test_sharp.jpg images/output/sharp_openmp.jpg sharpen

./convolution_cuda images/input/test_sharp.jpg images/output/sharp_cuda.jpg sharpen
```

---

## Performance Comparison

**Test Environment:**
- **CPU:** 4 cores (WSL2 on Azure VM)
- **GPU:** NVIDIA Tesla T4 (2,560 CUDA cores, 16 GB VRAM)
- **OpenMP Threads:** 4 and 8 (oversubscribed)
- **MPI Processes:** 4 and 8 (oversubscribed)
- **POSIX Threads:** 4 and 8

### Execution Time (seconds)

| Filter   | Kernel Size | Serial   | POSIX (4T) | POSIX (8T) | OpenMP (4T) | OpenMP (8T) | MPI (4P) | MPI (8P) | CUDA (T4) |
|----------|-------------|----------|------------|------------|-------------|-------------|----------|----------|------------|
| Blur     | 21×21       | 75.2188  | 76.3906*   | 75.7812*   | 19.0556     | 22.8717     | 19.0895  | 18.9988  | 0.1115     |
| Edge     | 3×3         | 1.9531   | 2.2188*    | 2.1094*    | 0.8603      | 0.9165      | 0.5196   | 0.5093   | 0.0134     |
| Sharpen  | 3×3         | 0.2500   | 0.3125*    | 0.2656*    | 0.1412      | 0.9113      | 0.0932   | 0.4664   | 0.0044     |


### Speedup vs Serial (×)

| Filter   | OpenMP (4T) | OpenMP (8T) | MPI (4P) | MPI (8P) | CUDA     |
|----------|-------------|-------------|----------|----------|----------|
| Blur     | 3.95×       | 3.29×       | 3.94×    | 3.96×    | 674.6×   |
| Edge     | 2.27×       | 2.13×       | 3.76×    | 3.83×    | 145.8×   |
| Sharpen  | 1.77×       | 0.27×       | 2.68×    | 0.54×    | 56.8×    |

### Key Observations

1. **CUDA dominates on large workloads** — For the blur filter (21×21 kernel), CUDA is **~675× faster** than serial, processing the image in just 0.11 seconds vs 75 seconds.

2. **OpenMP and MPI perform similarly** — Both achieve ~4× speedup on blur with 4 cores/processes, which is near the theoretical maximum for 4 parallel workers.

3. **Kernel size matters** — The blur filter (21×21 = 441 operations per pixel) benefits the most from parallelization. The sharpen filter (3×3 = 9 operations per pixel) has less work per pixel, so the overhead of parallelization reduces the relative speedup.

4. **GPU overhead is visible on small workloads** — CUDA's speedup drops from 675× (blur) to 57× (sharpen) because the data transfer between CPU and GPU becomes a larger fraction of total time when the computation per pixel is small.

5. **MPI edges out OpenMP on small kernels** — For edge detection and sharpen, MPI slightly outperforms OpenMP, likely due to differences in scheduling and memory access patterns.

6. **POSIX threads show similar CPU time to serial** — The POSIX version uses `clock()` which sums CPU time across all threads. The reported ~76s for blur with 4 threads doesn't mean it took 76s on the wall clock — each thread used ~19s of CPU time, totaling ~76s. The actual wall-clock time would be ~19s, comparable to OpenMP.
