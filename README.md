# Parallel Image Convolution

Image convolution implemented using multiple parallelization approaches — **Serial**, **OpenMP**, **POSIX Threads**, **MPI**, **CUDA**, and **Hybrid (MPI+OpenMP)** — to compare performance across CPU and GPU. Each version applies a convolution kernel (filter) to an input image pixel-by-pixel and writes the result to an output image.

## Project Structure

```
project/
|
+-- src/
|   +-- image_utils.c              # Image load/save using stb_image
|   |
|   +-- serial/
|   |   +-- convolution_serial.c   # Single-threaded CPU implementation
|   |
|   +-- openmp/
|   |   +-- convolution_openmp.c   # Multi-threaded CPU with OpenMP
|   |
|   +-- posix/
|   |   +-- convolution_posix.c    # Multi-threaded CPU with POSIX threads
|   |
|   +-- mpi/
|   |   +-- convolution_mpi.c      # Distributed-memory parallel with MPI
|   |
|   +-- hybrid/
|   |   +-- hybrid_mpi_openmp.c    # Hybrid MPI + OpenMP implementation
|   |
|   +-- cuda/
|       +-- convolution_cuda.cu    # GPU implementation using CUDA
|
+-- include/
|   +-- image_utils.h              # Image struct and function declarations
|   +-- stb_image.h                # stb image loading library
|   +-- stb_image_write.h          # stb image writing library
|
+-- images/
    +-- input/                     # Place input images here
    |   +-- test.jpg               # Input image for blur
    |   +-- test_edge.jpg          # Input image for edge detection
    |   +-- test_sharp.jpg         # Input image for sharpening
    |
    +-- output/                    # Processed images saved here
```

## Supported Filters

| Filter     | Description                                      | Kernel Size |
|------------|--------------------------------------------------|-------------|
| `blur`     | Gaussian blur (sigma=7.0)                        | 21x21       |
| `edge`     | Edge detection (Laplacian)                        | 3x3         |
| `sharpen`  | Sharpening filter                                 | 3x3         |

## Prerequisites

- **GCC** (for Serial, OpenMP, and POSIX builds)
- **MPI** (Microsoft MPI on Windows, OpenMPI on Linux)
- **NVIDIA CUDA Toolkit** (for CUDA build)
- **MSVC (cl.exe)** — required by nvcc on Windows

## Compilation and Running

All commands assume you are in the project root directory.

---

### 1. Serial (Single-threaded CPU)

**Compile:**
```bash
gcc -o convolution_serial src/serial/convolution_serial.c src/image_utils.c -I include -lm
```

**Run:**
```bash
./convolution_serial images/input/test.jpg images/output/blur_serial.jpg blur
./convolution_serial images/input/test_edge.jpg images/output/edge_serial.jpg edge
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
./convolution_openmp images/input/test.jpg images/output/blur_openmp.jpg blur
./convolution_openmp images/input/test_edge.jpg images/output/edge_openmp.jpg edge
./convolution_openmp images/input/test_sharp.jpg images/output/sharp_openmp.jpg sharpen
```

Control threads with OMP_NUM_THREADS:
```bash
# Windows PowerShell
$env:OMP_NUM_THREADS = 4

# Linux/macOS
export OMP_NUM_THREADS=4
```

---

### 3. POSIX Threads (Shared-memory CPU)

**Compile:**
```bash
gcc -o convolution_pthreads src/posix/convolution_posix.c src/image_utils.c -I include -lpthread -lm
```

**Run:**
```bash
./convolution_pthreads images/input/test.jpg images/output/blur_posix.jpg blur 4
./convolution_pthreads images/input/test_edge.jpg images/output/edge_posix.jpg edge 4
./convolution_pthreads images/input/test_sharp.jpg images/output/sharp_posix.jpg sharpen 4
```

The 4th argument specifies the number of threads:
```bash
./convolution_pthreads images/input/test.jpg images/output/blur_posix.jpg blur 8
```

---

### 4. MPI (Distributed-memory CPU)

**Compile:**
```bash
mpicc -o convolution_mpi src/mpi/convolution_mpi.c src/image_utils.c -I include -lm
```

**Run (Windows — Microsoft MPI):**
```bash
mpiexec -n 4 ./convolution_mpi images/input/test.jpg images/output/blur_mpi.jpg blur
mpiexec -n 4 ./convolution_mpi images/input/test_edge.jpg images/output/edge_mpi.jpg edge
mpiexec -n 4 ./convolution_mpi images/input/test_sharp.jpg images/output/sharp_mpi.jpg sharpen
```

**Run (Linux — OpenMPI):**
```bash
mpirun -np 4 ./convolution_mpi images/input/test.jpg images/output/blur_mpi.jpg blur
mpirun -np 4 ./convolution_mpi images/input/test_edge.jpg images/output/edge_mpi.jpg edge
mpirun -np 4 ./convolution_mpi images/input/test_sharp.jpg images/output/sharp_mpi.jpg sharpen
```

---

### 5. Hybrid MPI + OpenMP (Distributed + Shared Memory)

Combines MPI for inter-process row distribution with OpenMP for intra-process thread-level parallelism. Each MPI rank spawns multiple OpenMP threads, giving hierarchical parallelism.

**Compile:**
```bash
mpicc -O2 -fopenmp -o convolution_hybrid src/hybrid/hybrid_mpi_openmp.c src/image_utils.c -I include -lm
```

**Set OpenMP threads per MPI rank:**
```bash
# Linux/macOS
export OMP_NUM_THREADS=4

# Windows PowerShell
$env:OMP_NUM_THREADS = 4
```

**Run (Linux — OpenMPI):**
```bash
mpirun -np 4 ./convolution_hybrid images/input/test.jpg images/output/blur_hybrid.jpg blur
mpirun -np 4 ./convolution_hybrid images/input/test_edge.jpg images/output/edge_hybrid.jpg edge
mpirun -np 4 ./convolution_hybrid images/input/test_sharp.jpg images/output/sharp_hybrid.jpg sharpen
```

**Run (Windows — Microsoft MPI):**
```bash
mpiexec -n 4 ./convolution_hybrid images/input/test.jpg images/output/blur_hybrid.jpg blur
mpiexec -n 4 ./convolution_hybrid images/input/test_edge.jpg images/output/edge_hybrid.jpg edge
mpiexec -n 4 ./convolution_hybrid images/input/test_sharp.jpg images/output/sharp_hybrid.jpg sharpen
```

**Expected output:**
```
Hybrid MPI+OpenMP convolution took : 0.2150 seconds
  MPI ranks                        : 4
  OpenMP threads per rank          : 4
  Total parallel workers           : 16
```

**Worker scaling guide:**

| MPI Ranks | OMP_NUM_THREADS | Total Workers |
|-----------|-----------------|---------------|
| 2         | 2               | 4             |
| 2         | 4               | 8             |
| 4         | 4               | 16            |
| 4         | 8               | 32            |

---

### 6. CUDA (GPU)

#### Windows Setup

nvcc requires MSVC (cl.exe). Add the MSVC directory to your PATH:
```
C:\Program Files\Microsoft Visual Studio\<version>\<edition>\VC\Tools\MSVC\<toolset>\bin\HostX64\x64
```

**Compile:**
```bash
nvcc -allow-unsupported-compiler -o convolution_cuda src/cuda/convolution_cuda.cu src/image_utils.c -I include
```

**Run:**
```bash
./convolution_cuda images/input/test.jpg images/output/blur_cuda.jpg blur
./convolution_cuda images/input/test_edge.jpg images/output/edge_cuda.jpg edge
./convolution_cuda images/input/test_sharp.jpg images/output/sharp_cuda.jpg sharpen
```

---

### Usage Summary

```
<executable> <input_image> <output_image> <filter_type> [num_threads]
```

| Parameter      | Description                                          |
|----------------|------------------------------------------------------|
| `input_image`  | Path to the input image (JPG, PNG, BMP, etc.)        |
| `output_image` | Path to save the filtered output image               |
| `filter_type`  | One of: blur, edge, sharpen                          |
| `num_threads`  | (POSIX only) Number of threads to use                |

---

## Performance Comparison

**Test Environment:**
- **CPU:** 4 cores (Azure VM, Windows)
- **GPU:** NVIDIA Tesla T4 (40 SMs, Compute Capability 7.5)
- **Compiler:** GCC 15.2.0 (MSYS2 MinGW-w64), NVCC (CUDA Toolkit)
- **MPI:** Microsoft MPI 10.1

---

### Serial Baseline (seconds)

| Filter   | Kernel Size | Time (s) |
|----------|-------------|-----------|
| Blur     | 21x21       | 80.7870   |
| Edge     | 3x3         | 2.0890    |
| Sharpen  | 3x3         | 0.2660    |

---

### OpenMP Thread Scaling (seconds)

| Filter   | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|----------|----------|-----------|-----------|-----------|
| Blur     | 80.7510  | 41.1900   | 21.3780   | 21.3920   |
| Edge     | 2.2980   | 1.3220    | 0.8140    | 0.7640    |
| Sharpen  | 0.3020   | 0.1630    | 0.1020    | 0.1010    |

### OpenMP Speedup vs 1 Thread

| Filter   | 2 Threads | 4 Threads | 8 Threads |
|----------|-----------|-----------|-----------|
| Blur     | 1.96x     | 3.78x     | 3.77x     |
| Edge     | 1.74x     | 2.82x     | 3.01x     |
| Sharpen  | 1.85x     | 2.96x     | 2.99x     |

---

### POSIX Thread Scaling (seconds)

| Filter   | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|----------|----------|-----------|-----------|-----------|
| Blur     | 81.7374  | 39.5633   | 21.3648   | 21.4025   |
| Edge     | 2.1323   | 1.0611    | 0.5457    | 0.6408    |
| Sharpen  | 0.2591   | 0.1348    | 0.0735    | 0.0772    |

### POSIX Speedup vs 1 Thread

| Filter   | 2 Threads | 4 Threads | 8 Threads |
|----------|-----------|-----------|-----------|
| Blur     | 2.07x     | 3.83x     | 3.82x     |
| Edge     | 2.01x     | 3.91x     | 3.33x     |
| Sharpen  | 1.92x     | 3.52x     | 3.36x     |

---

### MPI Process Scaling (seconds)

| Filter   | 1 Process | 2 Processes | 4 Processes | 8 Processes |
|----------|-----------|-------------|-------------|-------------|
| Blur     | 81.4596   | 40.6525     | 21.7136     | 21.5084     |
| Edge     | 2.2052    | 1.1689      | 0.5519      | 0.5860      |
| Sharpen  | 0.2625    | 0.1424      | 0.0704      | 0.0709      |

### MPI Speedup vs 1 Process

| Filter   | 2 Processes | 4 Processes | 8 Processes |
|----------|-------------|-------------|-------------|
| Blur     | 2.00x       | 3.75x       | 3.79x       |
| Edge     | 1.89x       | 3.99x       | 3.76x       |
| Sharpen  | 1.84x       | 3.73x       | 3.70x       |

---

### Comparison at 4 Workers (seconds)

| Filter   | Serial   | OpenMP (4T) | POSIX (4T) | MPI (4P) | Hybrid (2Rx2T) | CUDA (GPU) |
|----------|----------|-------------|------------|----------|----------------|------------|
| Blur     | 80.7870  | 21.3780     | 21.3648    | 21.7136  | 2.3592         | 0.0510     |
| Edge     | 2.0890   | 0.8140      | 0.5457     | 0.5519   | 0.1160         | 0.0114     |
| Sharpen  | 0.2660   | 0.1020      | 0.0735     | 0.0704   | 0.1039         | 0.0039     |

### Speedup vs Serial — 4 Workers

| Filter   | OpenMP (4T) | POSIX (4T) | MPI (4P) | Hybrid (2Rx2T) | CUDA (GPU) |
|----------|-------------|------------|----------|----------------|------------|
| Blur     | 3.78x       | 3.78x      | 3.72x    | 34.24x         | 1584x      |
| Edge     | 2.57x       | 3.83x      | 3.78x    | 18.01x         | 183x       |
| Sharpen  | 2.61x       | 3.62x      | 3.78x    | 2.56x          | 68x        |

---

### CUDA GPU Performance (seconds)

**GPU:** NVIDIA Tesla T4 (40 SMs, Compute Capability 7.5)

| Filter   | Time (s) | Speedup vs Serial | Speedup vs Best CPU (4 workers) |
|----------|----------|-------------------|---------------------------------|
| Blur     | 0.0510   | 1584x             | 419x (vs POSIX 4T)              |
| Edge     | 0.0114   | 183x              | 48x (vs POSIX 4T)               |
| Sharpen  | 0.0039   | 68x               | 18x (vs MPI 4P)                 |

---

### Key Observations

1. **Linear scaling up to physical core count** — All three CPU-parallel approaches (OpenMP, MPI, POSIX) achieve near-linear speedup up to 4 threads/processes (matching the 4 physical CPU cores). Blur with 4 workers is ~3.7-3.8x faster than serial.

2. **8 workers = no gain beyond 4 cores** — Doubling from 4 to 8 threads/processes gives virtually zero improvement (e.g., blur: 21.38s to 21.39s for OpenMP, 21.36s to 21.40s for POSIX, 21.71s to 21.51s for MPI). Extra workers compete for the same 4 physical cores.

3. **Kernel size determines parallelization benefit** — The 21x21 blur kernel (441 operations per pixel) shows the best absolute time savings because there is enough work to keep all cores busy. The 3x3 kernels have less compute per pixel, so thread management overhead reduces relative speedup.

4. **POSIX threads and MPI outperform OpenMP on small kernels** — For edge detection, POSIX (0.546s) and MPI (0.552s) significantly beat OpenMP (0.814s) at 4 workers. This suggests OpenMP's dynamic scheduling has higher overhead for fast operations.

5. **MPI achieves competitive scaling on small kernels** — For sharpen at 4 processes, MPI (0.0704s) and POSIX (0.0735s) both outperform OpenMP (0.1020s). MPI's separate memory spaces help avoid cache contention between workers.

6. **2 threads/processes achieve near-perfect 2x speedup** — On blur, all approaches hit ~2.0x with 2 workers (OpenMP 1.96x, POSIX 2.07x, MPI 2.00x), showing minimal overhead when parallelism is moderate.

7. **POSIX shows slight regression at 8 threads for edge** — Edge goes from 0.546s (4T) to 0.641s (8T), a 17% slowdown. The workload is too small for 8 threads, and thread creation/synchronization overhead begins to dominate.

8. **All approaches converge on large workloads** — For the compute-heavy blur filter, all three approaches achieve nearly identical performance at 4 workers (~21.3-21.7s), since the large per-pixel workload dominates over scheduling and communication overhead.

9. **POSIX vs OpenMP trade-off** — POSIX threads offer better raw performance on small kernels but require manual thread management (create, join, divide work). OpenMP provides competitive speedup on large kernels with just a single pragma directive, making it far more maintainable.

10. **CUDA delivers orders-of-magnitude speedup** — The Tesla T4 GPU achieves 1584x speedup over serial for blur (0.051s vs 80.79s) and 419x over the best 4-core CPU result. The massively parallel architecture (thousands of concurrent threads) is ideal for image convolution where each pixel is independent.

11. **GPU benefit scales with kernel size** — Blur (21x21 kernel) sees 1584x speedup, edge (3x3) sees 183x, and sharpen (3x3 on smaller image) sees 68x. Larger kernels mean more compute per pixel, better utilizing the GPU's ALUs and hiding memory latency.

12. **Consistent results across runs** — All implementations show stable, reproducible performance with minimal variance between runs, confirming the benchmarks are reliable on this 4-core Azure VM with Tesla T4 GPU environment.

13. **Hybrid MPI+OpenMP enables hierarchical parallelism** — By combining MPI for inter-process row distribution with OpenMP for intra-process thread parallelism, the hybrid approach can exploit both distributed and shared memory simultaneously. On a multi-node cluster, this allows scaling beyond a single machine's core count while keeping communication overhead low.

---

## Source Code Details

### Serial Implementation (src/serial/convolution_serial.c)
- Single-threaded pixel-by-pixel convolution
- Uses clock() for timing
- Baseline for all speedup comparisons

### OpenMP Implementation (src/openmp/convolution_openmp.c)
- Uses `#pragma omp parallel for collapse(2) schedule(dynamic) shared(input, output, kernel, kernel_size)` to parallelize the nested pixel loops
- Explicit data scoping with shared clause (per LLNL HPC tutorial best practices)
- Uses omp_get_wtime() for wall-clock timing
- Thread count controlled via OMP_NUM_THREADS environment variable

### POSIX Implementation (src/posix/convolution_posix.c)
- Manually creates threads with pthread_create / pthread_join
- Uses pthread_attr_t with PTHREAD_CREATE_JOINABLE for explicit thread attributes
- Error checking on thread creation and join operations
- Divides image rows evenly among threads
- Uses clock_gettime(CLOCK_MONOTONIC) for accurate wall-clock timing
- Thread count passed as 4th command-line argument

### MPI Implementation (src/mpi/convolution_mpi.c)
- Uses MPI_Scatter to distribute image rows and MPI_Gather to collect results
- Each process computes convolution on its assigned rows
- Uses MPI_Bcast for metadata distribution and MPI_Barrier for synchronization
- Uses MPI_Wtime() for timing
- Process count controlled via mpiexec -n (Windows) or mpirun -np (Linux)

### Hybrid MPI+OpenMP Implementation (src/hybrid/hybrid_mpi_openmp.c)
- Uses MPI_Init_thread with MPI_THREAD_FUNNELED so only the main thread calls MPI while OpenMP threads run freely inside each rank
- MPI_Bcast distributes image data and kernel to all ranks
- Each rank receives a contiguous block of rows based on rank ID and total process count
- `#pragma omp parallel for collapse(2) schedule(dynamic, 4)` parallelises the pixel loop within each rank
- apply_kernel() is thread-safe — reads only from shared read-only buffers
- MPI_Gatherv collects variable-length results from each rank back to rank 0
- MPI_Reduce with MPI_MAX reports the true wall-clock bottleneck across all ranks
- Reports MPI ranks, OpenMP threads per rank, and total parallel workers
- Thread count controlled via OMP_NUM_THREADS; process count via mpirun -np or mpiexec -n

### CUDA Implementation (src/cuda/convolution_cuda.cu)
- GPU-accelerated convolution using CUDA kernels (NVIDIA CUDA Programming Guide)
- **Constant memory** (`__constant__`) for convolution kernel — cached and broadcast to all threads in a warp
- **Shared memory tiling** (`__shared__`) — thread blocks cooperatively load image tiles including halo region, reducing redundant global memory reads
- Each CUDA thread processes one output pixel across all channels
- Uses `cudaMemcpyToSymbol` to transfer kernel to constant memory
- 16x16 thread blocks with dynamic shared memory allocation
- CUDA Events (`cudaEventRecord`) for GPU-accurate timing
- Full error checking with `cudaGetLastError` and `cudaDeviceSynchronize`
- Prints device properties (compute capability, multiprocessors, shared/constant memory)