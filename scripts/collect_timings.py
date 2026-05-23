#!/usr/bin/env python3
"""
collect_timings.py

Run the project binaries (user must compile them first), parse timing output,
and write CSV files under report/data/ for PGFPlots consumption.

Usage:
  python scripts/collect_timings.py
  python scripts/collect_timings.py --workers 1 2 4 6 8
  python scripts/collect_timings.py --filter blur
  python scripts/collect_timings.py --dry
"""
import csv
import subprocess
import re
import argparse
import os
import sys

# Binaries are in project root, script is in scripts/ — so go one level up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'report', 'data')

# Correct input images per filter (3 separate images as seen in file structure)
INPUT_IMAGES = {
    'blur':    'images/input/test.jpg',
    'edge':    'images/input/test_edge.jpg',
    'sharpen': 'images/input/test_sharp.jpg',
}

OUTPUT_IMAGES = {
    'blur':    'images/output/blur_{impl}.jpg',
    'edge':    'images/output/edge_{impl}.jpg',
    'sharpen': 'images/output/sharp_{impl}.jpg',
}

TIME_RE = re.compile(r"([0-9]+\.?[0-9]*)\s*seconds")

def parse_time(output):
    for line in output.splitlines():
        if 'took' in line.lower():
            m = TIME_RE.search(line)
            if m:
                return float(m.group(1))
    return None

def run_command(cmd, timeout=None):
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT   # run from project root so relative paths work
        )
        return p.stdout
    except Exception as e:
        return str(e)

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows, header):
    ensure_outdir(os.path.dirname(path))
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)

def pivot(raw_rows, implementations, workers_list):
    table = []
    for w in workers_list:
        row = [w]
        for impl in implementations:
            val = ''
            for r in raw_rows:
                if r[0] == impl and int(r[1]) == int(w):
                    val = r[2]
                    break
            row.append(val)
        table.append(row)
    return table

def build_commands(filter_type, n):
    """
    Build the command list for each implementation.
    Binary names and paths corrected to match actual project structure.
    """
    inp = INPUT_IMAGES[filter_type]
    commands = []

    # Serial — binary in project root
    commands.append((
        'Serial', n,
        ['./convolution_serial', inp,
         OUTPUT_IMAGES[filter_type].format(impl='serial'), filter_type]
    ))

    # OpenMP — binary in project root, threads via env variable (set before calling)
    commands.append((
        'OpenMP', n,
        ['./convolution_openmp', inp,
         OUTPUT_IMAGES[filter_type].format(impl='openmp'), filter_type]
    ))

    # POSIX — binary named convolution_pthreads (NOT convolution_posix)
    # thread count passed as 4th argument
    commands.append((
        'POSIX', n,
        ['./convolution_pthreads', inp,
         OUTPUT_IMAGES[filter_type].format(impl='posix'), filter_type, str(n)]
    ))

    # MPI — binary in project root
    commands.append((
        'MPI', n,
        ['mpiexec', '-n', str(n), './convolution_mpi', inp,
         OUTPUT_IMAGES[filter_type].format(impl='mpi'), filter_type]
    ))

    # CUDA — binary in project root, no worker count argument
    commands.append((
        'CUDA', n,
        ['./convolution_cuda', inp,
         OUTPUT_IMAGES[filter_type].format(impl='cuda'), filter_type]
    ))

    # Hybrid — binary in project root, MPI processes × OMP threads
    # e.g. n=4 → 2 MPI processes × 2 OMP threads
    mpi_procs = max(1, n // 2)
    omp_threads = max(1, n // mpi_procs)
    commands.append((
        'Hybrid', n,
        ['mpiexec', '-n', str(mpi_procs), './convolution_hybrid', inp,
         OUTPUT_IMAGES[filter_type].format(impl='hybrid'), filter_type]
    ))

    return commands

def set_omp_threads(n):
    """Set OMP_NUM_THREADS in the environment for OpenMP and Hybrid runs."""
    os.environ['OMP_NUM_THREADS'] = str(n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers', nargs='+', default=['1', '2', '4', '8'])
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--filter', choices=['blur', 'edge', 'sharpen', 'all'],
                        default='all', help='Which filter to benchmark')
    parser.add_argument('--dry', action='store_true',
                        help="Don't run commands, just generate CSV headers")
    args = parser.parse_args()

    filters = ['blur', 'edge', 'sharpen'] if args.filter == 'all' else [args.filter]

    # Ensure output image directory exists
    ensure_outdir(os.path.join(PROJECT_ROOT, 'images', 'output'))

    all_raw_rows = []   # (filter, impl, workers, time)

    if args.dry:
        write_csv(os.path.join(args.outdir, 'raw_timings.csv'), [],
                  ['Filter', 'Implementation', 'Workers', 'Time'])
        print('Dry run: generated CSV headers in', args.outdir)
        return

    for filter_type in filters:
        print(f'\n=== Filter: {filter_type} ===')
        raw_rows = []

        for w in args.workers:
            n = int(w)

            # Set OMP_NUM_THREADS for OpenMP/Hybrid before running
            set_omp_threads(n)

            commands = build_commands(filter_type, n)

            for impl, workers, cmd in commands:
                # CUDA has no worker scaling — skip duplicate runs
                if impl == 'CUDA' and n > 1:
                    continue

                print(f'  Running: {impl} workers={n} -> {" ".join(cmd)}')
                out = run_command(cmd, timeout=args.timeout)
                t = parse_time(out)

                if t is None:
                    print(f'  Warning: could not parse time from {impl} output')
                    logf = os.path.join(args.outdir, f'log_{filter_type}_{impl}_{n}.txt')
                    ensure_outdir(os.path.dirname(logf))
                    with open(logf, 'w') as lf:
                        lf.write(out)

                raw_rows.append((impl, n, t if t is not None else ''))
                all_raw_rows.append((filter_type, impl, n, t if t is not None else ''))

        # Write per-filter CSVs
        implementations = ['Serial', 'OpenMP', 'POSIX', 'MPI', 'CUDA', 'Hybrid']
        pivot_table = pivot(raw_rows, implementations, args.workers)

        raw_path = os.path.join(args.outdir, f'raw_{filter_type}.csv')
        pivot_path = os.path.join(args.outdir, f'times_{filter_type}_pivot.csv')

        write_csv(raw_path, raw_rows, ['Implementation', 'Workers', 'Time'])
        write_csv(pivot_path, pivot_table, ['Workers'] + implementations)

        print(f'  Wrote: {raw_path}')
        print(f'  Wrote: {pivot_path}')

    # Write combined CSV across all filters
    combined_path = os.path.join(args.outdir, 'raw_timings_all.csv')
    write_csv(combined_path, all_raw_rows, ['Filter', 'Implementation', 'Workers', 'Time'])
    print(f'\nWrote combined: {combined_path}')

if __name__ == '__main__':
    main()