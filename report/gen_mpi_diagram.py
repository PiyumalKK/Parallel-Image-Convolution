import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(-1.5, 16)
ax.set_ylim(-5.5, 2.2)
ax.set_aspect('equal')
ax.axis('off')

# Phase definitions: (x_start, width, label, gray_level)
phases = [
    (1.0, 2.2, 'Bcast', 0.88),
    (3.6, 2.4, 'Scatterv', 0.78),
    (6.4, 3.8, 'Local Compute', 0.95),
    (10.6, 1.6, 'Barrier', 0.65),
    (12.6, 2.4, 'Gatherv', 0.85),
]

row_h = 0.7
row_sep = 1.3
ranks = 4

# Phase labels at top
for (x, w, label, _) in phases:
    ax.text(x + w/2, 1.6, label, ha='center', va='center',
            fontsize=11, fontweight='bold')

# Time arrow
ax.annotate('', xy=(15.5, 1.0), xytext=(0.5, 1.0),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(15.7, 1.0, 'time', fontsize=11, fontweight='bold', va='center')

# Draw boxes for each rank
boxes = {}  # boxes[(rank, phase_idx)] = (x_center, y_center, x_start, x_end)

for r in range(ranks):
    y = -r * row_sep
    ax.text(-0.2, y, f'Rank {r}', ha='right', va='center',
            fontsize=12, fontweight='bold')
    # Light horizontal guide
    ax.plot([0.5, 15.3], [y, y], color='gray', lw=0.3, zorder=0)

    for pi, (x, w, label, gray) in enumerate(phases):
        rect = mpatches.FancyBboxPatch(
            (x, y - row_h/2), w, row_h,
            boxstyle="round,pad=0.05",
            facecolor=str(gray), edgecolor='black', linewidth=1.2,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)
        boxes[(r, pi)] = (x + w/2, y, x, x + w)

# Arrow helper
def draw_arrow(x1, y1, x2, y2, color='black', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, lw=lw, color=color),
                zorder=5)

# Bcast arrows: Rank 0 -> Rank 1,2,3 (staggered x offsets to avoid overlap)
offsets_b = [-0.3, 0.0, 0.3]
for i, r in enumerate(range(1, 4)):
    x0 = boxes[(0, 0)][0] + offsets_b[i]
    y0 = boxes[(0, 0)][1] - row_h/2
    y1 = boxes[(r, 0)][1] + row_h/2
    draw_arrow(x0, y0, x0, y1)

# Scatterv arrows: Rank 0 -> Rank 1,2,3
offsets_s = [-0.3, 0.0, 0.3]
for i, r in enumerate(range(1, 4)):
    x0 = boxes[(0, 1)][0] + offsets_s[i]
    y0 = boxes[(0, 1)][1] - row_h/2
    y1 = boxes[(r, 1)][1] + row_h/2
    draw_arrow(x0, y0, x0, y1)

# Barrier: dashed lines between consecutive ranks
for r in range(3):
    x0 = boxes[(r, 3)][0]
    y0 = boxes[(r, 3)][1] - row_h/2
    y1 = boxes[(r+1, 3)][1] + row_h/2
    ax.plot([x0, x0], [y0, y1], 'k--', lw=2, zorder=5)

# Gatherv arrows: Rank 1,2,3 -> Rank 0 (staggered)
offsets_g = [-0.3, 0.0, 0.3]
for i, r in enumerate(range(1, 4)):
    x0 = boxes[(r, 4)][0] + offsets_g[i]
    y0 = boxes[(r, 4)][1] + row_h/2
    y1 = boxes[(0, 4)][1] - row_h/2
    draw_arrow(x0, y0, x0, y1)

plt.tight_layout()
plt.savefig('images/mpi_timeline.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved images/mpi_timeline.png")
