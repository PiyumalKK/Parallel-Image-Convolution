#!/usr/bin/env python3
"""
collect_timings.py

Run the project binaries (user must compile them first), parse timing output,
and write CSV files under report/data/ for PGFPlots consumption.

Usage:
  python scripts/collect_timings.py --config config.py

The script expects each command to print a line containing the word "took"
and the number of seconds (e.g. "Serial convolution took: 80.7870 seconds").
"""
import csv
import subprocess
import re
import argparse
import os
import sys

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'data')

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
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
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
    # raw_rows: list of (impl, workers, time)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--workers', nargs='+', default=['1','2','4','8'])
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--dry', action='store_true', help='Don\'t run commands, just generate CSV headers')
    parser.add_argument('--config', default=None, help='Optional python config file that defines COMMANDS list')
    args = parser.parse_args()

    # Default commands; users should edit this or provide a config file.
    # Each entry: (ImplementationName, Workers, CommandList)
    COMMANDS = [
        ('Serial', 1, ['./build/convolution_serial', 'input/image.png', 'output/serial.png', 'blur']),
        ('OpenMP', 1, ['./build/convolution_openmp', 'input/image.png', 'output/openmp.png', 'blur']),
        ('POSIX', 1, ['./build/convolution_posix', 'input/image.png', 'output/posix.png', 'blur']),
        ('MPI', 1, ['mpiexec', '-n', '1', './build/convolution_mpi', 'input/image.png', 'output/mpi.png', 'blur']),
        ('CUDA', 1, ['./build/convolution_cuda', 'input/image.png', 'output/cuda.png', 'blur']),
    ]

    if args.config:
        cfg_path = os.path.abspath(args.config)
        if os.path.exists(cfg_path):
            ns = {}
            with open(cfg_path, 'r') as f:
                exec(f.read(), ns)
            if 'COMMANDS' in ns:
                COMMANDS = ns['COMMANDS']
        else:
            print('Config file not found:', cfg_path)
            sys.exit(1)

    raw_rows = []

    # If dry-run, just create headers and exit
    if args.dry:
        write_csv(os.path.join(args.outdir, 'raw_timings.csv'), [], ['Implementation','Workers','Time'])
        write_csv(os.path.join(args.outdir, 'times_pivot.csv'), [], ['Workers'] + [c[0] for c in COMMANDS])
        print('Dry run: generated CSV headers in', args.outdir)
        return

    for impl, base_workers, cmd_template in COMMANDS:
        for w in args.workers:
            # Build command: replace placeholder {n} if present else insert mpiexec for parallel runs
            try:
                n = int(w)
            except:
                n = base_workers

            cmd = list(cmd_template)
            # If command uses 'mpiexec' and has -n placeholder, user should provide in config
            # Common pattern: replace any occurrence of '{n}'
            cmd = [str(x).replace('{n}', str(n)) for x in cmd]

            print('Running:', impl, 'workers=', n, '->', ' '.join(cmd))
            out = run_command(cmd, timeout=args.timeout)
            t = parse_time(out)
            if t is None:
                print('Warning: could not parse time from output; saving raw output to log')
                logf = os.path.join(args.outdir, f'log_{impl}_{n}.txt')
                ensure_outdir(os.path.dirname(logf))
                with open(logf, 'w') as lf:
                    lf.write(out)
            raw_rows.append((impl, n, t if t is not None else ''))

    # Write raw CSV
    raw_path = os.path.join(args.outdir, 'raw_timings.csv')
    write_csv(raw_path, raw_rows, ['Implementation','Workers','Time'])

    # Pivot into times_pivot.csv
    implementations = [c[0] for c in COMMANDS]
    pivot_table = pivot(raw_rows, implementations, args.workers)
    pivot_path = os.path.join(args.outdir, 'times_pivot.csv')
    header = ['Workers'] + implementations
    write_csv(pivot_path, pivot_table, header)

    print('Wrote:', raw_path)
    print('Wrote:', pivot_path)

if __name__ == '__main__':
    main()
