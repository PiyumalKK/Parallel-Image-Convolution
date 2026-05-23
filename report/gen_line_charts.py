import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark_results.txt
workers = [1, 2, 4, 8]

# === BLUR (21x21, 3840x2160) ===
serial_blur = 81.426

omp_blur = [81.19, 40.457, 20.614, 20.584]
posix_blur = [81.6829, 39.666, 20.1618, 20.045]
mpi_blur = [80.1861, 39.8917, 20.2369, 20.4954]

# === EDGE (3x3, 3840x2160) ===
serial_edge = 2.123

omp_edge = [2.311, 1.290, 0.855, 0.762]
posix_edge = [2.1432, 1.0638, 0.5189, 0.5715]
mpi_edge = [2.0781, 1.0535, 0.533, 0.5656]

# === SHARPEN (3x3, 1252x896) ===
serial_sharpen = 0.263

omp_sharpen = [0.293, 0.165, 0.099, 0.100]
posix_sharpen = [0.2629, 0.1332, 0.0675, 0.0675]
mpi_sharpen = [0.2567, 0.1298, 0.068, 0.0811]

# === HYBRID (blur only, for hybrid chart) ===
hybrid_configs = ['Pure\nOMP-4', 'H 1x4', 'H 2x2', 'H 4x1', 'Pure\nOMP-8', 'H 1x8', 'H 2x4', 'H 4x2', 'Pure\nMPI-8']
hybrid_times = [20.614, 4.6255, 4.58, 4.78, 20.584, 4.7844, 4.6363, 4.4891, 20.4954]
# For H 2x2: from report table = 4.58, H 4x1: from report = 4.78

# Style
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})
markers = ['o', 's', '^']
colors = ['#228B22', '#D2691E', '#9400D3']  # green, orange, purple
labels = ['OpenMP', 'POSIX', 'MPI']

def plot_time(title, filename, serial, omp, posix, mpi, ylabel='Execution Time (s)'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for data, color, marker, label in zip([omp, posix, mpi], colors, markers, labels):
        ax.plot(workers, data, color=color, marker=marker, linewidth=2, markersize=7, label=label)
    ax.axhline(y=serial, color='#4682B4', linestyle='--', linewidth=1.5, label='Serial')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(workers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved figures/{filename}')

def plot_speedup(title, filename, serial, omp, posix, mpi):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for data, color, marker, label in zip([omp, posix, mpi], colors, markers, labels):
        speedups = [serial / t for t in data]
        ax.plot(workers, speedups, color=color, marker=marker, linewidth=2, markersize=7, label=label)
    # Ideal linear
    ax.plot(workers, workers, color='gray', linestyle=':', linewidth=1.5, label='Ideal linear')
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(title)
    ax.set_xticks(workers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved figures/{filename}')

# Generate all charts
print('Generating line charts...')

# Blur
plot_time('Gaussian Blur (21x21) - Execution Time', 'blur_time.png',
          serial_blur, omp_blur, posix_blur, mpi_blur)
plot_speedup('Gaussian Blur (21x21) - Speedup', 'blur_speedup.png',
             serial_blur, omp_blur, posix_blur, mpi_blur)

# Edge
plot_time('Edge Detection (3x3) - Execution Time', 'edge_time.png',
          serial_edge, omp_edge, posix_edge, mpi_edge)
plot_speedup('Edge Detection (3x3) - Speedup', 'edge_speedup.png',
             serial_edge, omp_edge, posix_edge, mpi_edge)

# Sharpen
plot_time('Sharpen (3x3) - Execution Time', 'sharpen_time.png',
          serial_sharpen, omp_sharpen, posix_sharpen, mpi_sharpen)
plot_speedup('Sharpen (3x3) - Speedup', 'sharpen_speedup.png',
             serial_sharpen, omp_sharpen, posix_sharpen, mpi_sharpen)

# Hybrid bar (keep as grouped bar since it's categorical configs)
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(hybrid_configs))
bars = ax.bar(x, hybrid_times, color=['#228B22', '#008B8B', '#008B8B', '#008B8B',
                                       '#228B22', '#008B8B', '#008B8B', '#008B8B', '#9400D3'],
              edgecolor='black', linewidth=0.5)
ax.set_xlabel('Configuration')
ax.set_ylabel('Execution Time (s)')
ax.set_title('Hybrid vs Pure - Gaussian Blur Execution Time')
ax.set_xticks(x)
ax.set_xticklabels(hybrid_configs, fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('figures/hybrid_bar.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved figures/hybrid_bar.png')

print('Done!')
