# run_all_benchmarks.ps1
# Comprehensive benchmark script for Parallel Image Convolution
# Collects timing + RMSE data for the report
# Azure VM: 4 physical cores, Tesla T4 GPU

Set-Location "C:\Users\Piyumal\Desktop\HPC\Parallel-Image-Convolution"

$outFile = "benchmark_results.txt"
"" | Out-File $outFile
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

function Log($msg) {
    Write-Host $msg
    $msg | Out-File $outFile -Append
}

Log "=============================================="
Log " BENCHMARK RESULTS - $timestamp"
Log " Machine: Azure VM (4 cores, Tesla T4)"
Log "=============================================="
Log ""

# ─── Step 0: Add MSVC to PATH for CUDA compilation ──────────────────────────
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + $env:PATH

# ─── Step 1: Compile hybrid + RMSE utility ───────────────────────────────────
Log "=== COMPILATION ==="

# Compile RMSE compare tool
Log "Compiling rmse_compare..."
gcc -O2 -o rmse_compare.exe src/rmse_compare.c src/image_utils.c -I include -lm
if ($LASTEXITCODE -ne 0) { Log "ERROR: rmse_compare compilation failed"; exit 1 }

# Compile hybrid (MPI+OpenMP)
Log "Compiling hybrid_mpi_openmp..."
mpicc -fopenmp -O2 -o convolution_hybrid.exe src/hybrid/hybrid_mpi_openmp.c src/image_utils.c -I include -lm
if ($LASTEXITCODE -ne 0) { Log "ERROR: hybrid compilation failed"; exit 1 }

Log "All compilations successful."
Log ""

# ─── Step 2: Define test cases ───────────────────────────────────────────────
$filters = @(
    @{ name="blur";    input="images/input/test.jpg";       serial_out="images/output/blur_serial.jpg" },
    @{ name="edge";    input="images/input/test_edge.jpg";  serial_out="images/output/edge_serial.jpg" },
    @{ name="sharpen"; input="images/input/test_sharp.jpg"; serial_out="images/output/sharp_serial.jpg" }
)

$threadCounts = @(1, 2, 4, 8)

# ─── Step 3: Run Serial baseline ────────────────────────────────────────────
Log "=== SERIAL BASELINE ==="
foreach ($f in $filters) {
    $result = & ./convolution_serial.exe $f.input $f.serial_out $f.name 2>&1
    Log "  $($f.name): $result"
}
Log ""

# ─── Step 4: Run OpenMP ─────────────────────────────────────────────────────
Log "=== OPENMP ==="
foreach ($t in $threadCounts) {
    $env:OMP_NUM_THREADS = $t
    Log "  Threads=$t"
    foreach ($f in $filters) {
        $outImg = "images/output/$($f.name)_openmp_t$t.jpg"
        $result = & ./convolution_openmp.exe $f.input $outImg $f.name 2>&1
        Log "    $($f.name): $result"
    }
}
Log ""

# ─── Step 5: Run POSIX Pthreads ─────────────────────────────────────────────
Log "=== POSIX PTHREADS ==="
foreach ($t in $threadCounts) {
    Log "  Threads=$t"
    foreach ($f in $filters) {
        $outImg = "images/output/$($f.name)_posix_t$t.jpg"
        $result = & ./convolution_pthreads.exe $f.input $outImg $f.name $t 2>&1
        Log "    $($f.name): $result"
    }
}
Log ""

# ─── Step 6: Run MPI ────────────────────────────────────────────────────────
Log "=== MPI ==="
foreach ($t in $threadCounts) {
    Log "  Processes=$t"
    foreach ($f in $filters) {
        $outImg = "images/output/$($f.name)_mpi_p$t.jpg"
        $result = & mpiexec -n $t ./convolution_mpi.exe $f.input $outImg $f.name 2>&1
        Log "    $($f.name): $($result -join ' ')"
    }
}
Log ""

# ─── Step 7: Run CUDA ───────────────────────────────────────────────────────
Log "=== CUDA ==="
foreach ($f in $filters) {
    $outImg = "images/output/$($f.name)_cuda.jpg"
    $result = & ./convolution_cuda.exe $f.input $outImg $f.name 2>&1
    Log "  $($f.name): $($result -join ' ')"
}
Log ""

# ─── Step 8: Run Hybrid MPI+OpenMP configurations ───────────────────────────
Log "=== HYBRID MPI+OpenMP ==="
$hybridConfigs = @(
    @{ procs=1; threads=4; label="1x4" },
    @{ procs=2; threads=2; label="2x2" },
    @{ procs=4; threads=1; label="4x1" },
    @{ procs=1; threads=8; label="1x8" },
    @{ procs=2; threads=4; label="2x4" },
    @{ procs=4; threads=2; label="4x2" }
)

foreach ($cfg in $hybridConfigs) {
    $env:OMP_NUM_THREADS = $cfg.threads
    Log "  Config: $($cfg.label) ($($cfg.procs)P x $($cfg.threads)T)"
    foreach ($f in $filters) {
        $outImg = "images/output/$($f.name)_hybrid_$($cfg.label).jpg"
        $result = & mpiexec -n $cfg.procs ./convolution_hybrid.exe $f.input $outImg $f.name 2>&1
        Log "    $($f.name): $($result -join ' ')"
    }
}
Log ""

# ─── Step 9: RMSE Comparison vs Serial ──────────────────────────────────────
Log "=== RMSE COMPARISON (vs Serial) ==="

# OpenMP at 4 threads
Log "  OpenMP (4 threads):"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_openmp_t4.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

# POSIX at 4 threads
Log "  POSIX (4 threads):"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_posix_t4.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

# MPI at 4 processes
Log "  MPI (4 processes):"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_mpi_p4.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

# CUDA
Log "  CUDA:"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_cuda.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

# Hybrid 2x2
Log "  Hybrid 2x2 (2P x 2T):"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_hybrid_2x2.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

# Hybrid 2x4
Log "  Hybrid 2x4 (2P x 4T):"
foreach ($f in $filters) {
    $parImg = "images/output/$($f.name)_hybrid_2x4.jpg"
    if (Test-Path $parImg) {
        $result = & ./rmse_compare.exe $f.serial_out $parImg 2>&1
        Log "    $($f.name): $result"
    }
}

Log ""
Log "=== DONE ==="
Log "Results saved to: $outFile"
Write-Host "`nAll benchmarks complete! Results in $outFile"
