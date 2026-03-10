# Parallel Image Convolution

Image convolution implemented using three parallelization approaches — **Serial**, **OpenMP**, and **CUDA** — to compare performance across CPU and GPU. Each version applies a convolution kernel (filter) to an input image pixel-by-pixel and writes the result to an output image.

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
    │   └── test.jpg
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

- **GCC** (for Serial and OpenMP builds)
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
./convolution_serial images/input/test.jpg images/output/result_serial.jpg blur
```

---

### 2. OpenMP (Multi-threaded CPU)

**Compile:**
```bash
gcc -fopenmp -o convolution_openmp src/openmp/convolution_openmp.c src/image_utils.c -I include -lm
```

**Run:**
```bash
./convolution_openmp images/input/test.jpg images/output/result_openmp.jpg blur
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
mpirun -np 4 ./convolution_mpi images/input/test.jpg images/output/result_mpi.jpg blur
```

You can control the number of process using the -np option:

```bash
# Linux/macOS
# Run with 4 processes
mpirun -np 4 ./convolution_mpi images/input/test.jpg images/output/result_mpi.jpg blur

# Run with 8 processes
mpirun -np 8 ./convolution_mpi images/input/test.jpg images/output/result_mpi.jpg blur
```

---

### 3. CUDA (GPU)

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
./convolution_cuda images/input/test.jpg images/output/result_cuda.jpg blur
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
./convolution_serial  images/input/test.jpg images/output/blur_serial.jpg  blur
./convolution_openmp  images/input/test.jpg images/output/edge_openmp.jpg  edge
./convolution_cuda    images/input/test.jpg images/output/sharp_cuda.jpg   sharpen
```
