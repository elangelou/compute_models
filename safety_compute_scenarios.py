"""
AI Safety Research Compute Intensiveness — Three Timeline Scenarios
Based on Epoch AI's effective compute framework:
  - Frontier training compute: ~3×10^25 FLOPs in 2024, growing ~2.5× per year
  - Algorithmic efficiency gains: 2× effective every 9 months
  - Research tasks scale sublinearly with frontier (exponents 0.15–0.80 by task type)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "axes.facecolor": "#f8f9fb",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Epoch framework constants ─────────────────────────────────────────────────
FRONTIER_2024 = 3e25          # FLOPs — GPT-4 / Gemini Ultra scale
FRONTIER_GROWTH = 2.5         # × per year (conservative; historical ~4×)
ALGO_DOUBLING_YRS = 9 / 12   # years for 2× algorithmic efficiency (Epoch: every 9 months)


def frontier(year, growth=FRONTIER_GROWTH):
    """Extrapolate frontier training compute from Epoch trends."""
    return FRONTIER_2024 * growth ** (year - 2024)


def frontier_long(year):
    """Long-timeline variant: growth slows to 1.4× after 2040 (hardware ceiling)."""
    if year <= 2040:
        return FRONTIER_2024 * 2.5 ** (year - 2024)
    f2040 = FRONTIER_2024 * 2.5 ** 16
    return f2040 * 1.4 ** (year - 2040)


def task_raw(base, exp, year, frontier_fn=frontier):
    """Raw FLOPs for a task: scales sublinearly with frontier compute."""
    return base * (frontier_fn(year) / FRONTIER_2024) ** exp


def algo_multiplier(year):
    """Cumulative algorithmic efficiency gain since 2024 (Epoch: 2× every 9 months)."""
    return 2 ** ((year - 2024) / ALGO_DOUBLING_YRS)


def task_effective(base, exp, year, frontier_fn=frontier):
    """Effective FLOPs = raw / algo_efficiency  — the 'real cost' after improvements."""
    return task_raw(base, exp, year, frontier_fn) / algo_multiplier(year)


def log_fmt(val, pos):
    exp = int(round(np.log10(val)))
    return f"$10^{{{exp}}}$"


fmt = mticker.FuncFormatter(log_fmt)

MARKER_EVERY = {200: 50, 300: 75, 400: 95, 500: 115}


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Short Timelines (AGI ≈ 2028)
# Tasks: Mechanistic Interpretability, Control AI, Red-teaming, Evaluations
# ══════════════════════════════════════════════════════════════════════════════
N1 = 300
years1 = np.linspace(2024, 2029, N1)
me = N1 // 4

fig, ax = plt.subplots(figsize=(11.5, 6.8))

# Frontier reference
ax.semilogy(years1, [frontier(y) for y in years1],
            "--", color="#37474F", lw=2.2, alpha=0.5,
            label="Frontier training compute (reference)")

# ── Task definitions: (label, base_2024_FLOPs, scale_exponent, color, marker) ──
# scale_exp intuition:
#   ~0.80 → closely tracks frontier (needs capable AI to test against)
#   ~0.65 → moderate scaling (inference-heavy but not training)
#   ~0.60 → lighter scaling (benchmark sweeps)
tasks1 = [
    ("Mechanistic Interpretability\n(Neel Nanda — circuit analysis,\nactivation patching on frontier models)",
     8e21, 0.72, "#E53935", "o"),
    ("Control AI agenda tasks\n(testing control protocols on\ncapable AI systems)",
     4e22, 0.80, "#FB8C00", "s"),
    ("Red-teaming\n(adversarial inference queries,\nvulnerability discovery)",
     2e21, 0.65, "#FDD835", "^"),
    ("Evaluations\n(benchmark suites, capability\nassessment at scale)",
     5e20, 0.60, "#43A047", "D"),
]

for label, base, exp, color, marker in tasks1:
    vals = [task_raw(base, exp, y) for y in years1]
    ax.semilogy(years1, vals, "-", color=color, lw=2.5, label=label,
                marker=marker, markevery=me, ms=8, markeredgewidth=0.6,
                markeredgecolor="white")

# AGI milestone
ax.axvline(2028, color="#B71C1C", lw=2.2, ls=":", alpha=0.9)
ax.text(2027.93, 3.5e29, "AGI ≈ 2028", color="#B71C1C", fontsize=10,
        fontweight="bold", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#B71C1C", alpha=0.9))
ax.axvspan(2026.8, 2028, alpha=0.055, color="#B71C1C", zorder=0,
           label="Critical compute window")

# Annotation: gap between frontier and tasks
ax.annotate("", xy=(2028, frontier(2028)), xytext=(2028, task_raw(4e22, 0.80, 2028)),
            arrowprops=dict(arrowstyle="<->", color="#607D8B", lw=1.4))
ax.text(2028.06, np.sqrt(frontier(2028) * task_raw(4e22, 0.80, 2028)),
        "Frontier ÷ task\n≈ 10³·⁵ gap", fontsize=8, color="#546E7A", va="center")

ax.set_xlim(2024, 2029.4)
ax.set_ylim(1e19, 2e30)
ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Compute per research campaign (FLOPs)", fontsize=12)
ax.set_title(
    "Scenario 1 — Short Timelines (AGI ≈ 2028)\n"
    "Compute Intensiveness of AI Safety Research Agendas",
    fontsize=13, fontweight="bold", pad=13,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.94, edgecolor="#ccc",
          labelspacing=0.6)
ax.text(0.99, 0.01,
        "Epoch AI effective compute framework | Frontier: ~2.5× yr⁻¹ from 2024 baseline (~3×10²⁵ FLOPs)\n"
        "Task compute scales sublinearly with frontier (exponents 0.60–0.80 by task type)",
        transform=ax.transAxes, fontsize=7.5, color="#888", ha="right", va="bottom")

plt.tight_layout()
plt.savefig("/Users/eleni/compute_models/scenario1_short_timelines.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scenario1_short_timelines.png")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Medium Timelines (AGI ≈ 2035)
# Tasks: Interpretability, Evaluations, Learning Theory
# + Effective compute (algo-efficiency-adjusted) shown as dashed variant
# ══════════════════════════════════════════════════════════════════════════════
N2 = 400
years2 = np.linspace(2024, 2036, N2)
me2 = N2 // 4

fig, ax = plt.subplots(figsize=(11.5, 6.8))

ax.semilogy(years2, [frontier(y) for y in years2],
            "--", color="#37474F", lw=2.2, alpha=0.5,
            label="Frontier training compute (reference)")

tasks2 = [
    ("Interpretability\n(systematic mechanistic analysis\nof large-scale models)",
     1e22, 0.75, "#7B1FA2", "o"),
    ("Evaluations\n(comprehensive capability & safety\nassessments at scale)",
     5e20, 0.62, "#0288D1", "D"),
    ("Learning Theory\n(empirical validation of theoretical\nresults on real architectures)",
     3e19, 0.47, "#00897B", "^"),
]

for label, base, exp, color, marker in tasks2:
    raw_vals = [task_raw(base, exp, y) for y in years2]
    eff_vals = [task_effective(base, exp, y) for y in years2]

    ax.semilogy(years2, raw_vals, "-", color=color, lw=2.5, label=f"{label} (raw)",
                marker=marker, markevery=me2, ms=8,
                markeredgewidth=0.6, markeredgecolor="white")
    ax.semilogy(years2, eff_vals, "--", color=color, lw=1.6, alpha=0.55,
                label=f"  └─ effective (÷ algo-efficiency)")

# AGI milestone
ax.axvline(2035, color="#B71C1C", lw=2.2, ls=":", alpha=0.9)
ax.text(2034.9, 5e37, "AGI ≈ 2035", color="#B71C1C", fontsize=10,
        fontweight="bold", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#B71C1C", alpha=0.9))
ax.axvspan(2033, 2035, alpha=0.055, color="#B71C1C", zorder=0,
           label="Critical compute window")

# Algo efficiency band annotation
mid_y = np.sqrt(task_raw(1e22, 0.75, 2032) * task_effective(1e22, 0.75, 2032))
ax.annotate("Algorithmic efficiency\n(÷ ~1,000× by 2032)", xy=(2032, task_effective(1e22, 0.75, 2032)),
            xytext=(2030.5, task_effective(1e22, 0.75, 2032) * 40),
            arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.3),
            fontsize=8.5, color="#6A1B9A")

# Manual legend entries for dashed lines
extra = [
    Line2D([0], [0], ls="--", lw=1.6, color="gray", alpha=0.7,
           label="Dashed = effective compute\n(÷ algorithmic efficiency gain,\n2× every 9 months per Epoch)"),
]
handles, labels = ax.get_legend_handles_labels()

ax.set_xlim(2024, 2037)
ax.set_ylim(5e16, 5e39)
ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Compute per research campaign (FLOPs)", fontsize=12)
ax.set_title(
    "Scenario 2 — Medium Timelines (AGI ≈ 2035)\n"
    "Compute Intensiveness of AI Safety Research Agendas",
    fontsize=13, fontweight="bold", pad=13,
)

# Clean legend: remove duplicate dashed entries, add single explanatory one
clean_handles = [h for h, l in zip(handles, labels) if "└─" not in l] + extra
clean_labels = [l for l in labels if "└─" not in l] + ["Dashed = effective compute\n(÷ algo-efficiency, 2× / 9 mo.)"]
ax.legend(clean_handles, clean_labels, loc="upper left", fontsize=9,
          framealpha=0.94, edgecolor="#ccc", labelspacing=0.55)

ax.text(0.99, 0.01,
        "Epoch AI effective compute framework | Frontier: ~2.5× yr⁻¹ | Algo efficiency: 2× / 9 months\n"
        "Dashed variants show effective compute — reduced cost after algorithmic improvements are applied",
        transform=ax.transAxes, fontsize=7.5, color="#888", ha="right", va="bottom")

plt.tight_layout()
plt.savefig("/Users/eleni/compute_models/scenario2_medium_timelines.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scenario2_medium_timelines.png")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Long Timelines (AGI ≈ 2055)
# Tasks: Learning Theory, Computational Complexity, Information Theory, Cryptography
# Frontier growth slows after 2040 (hardware/physics ceiling)
# ══════════════════════════════════════════════════════════════════════════════
N3 = 500
years3 = np.linspace(2024, 2056, N3)
me3 = N3 // 4

fig, ax = plt.subplots(figsize=(11.5, 6.8))

ax.semilogy(years3, [frontier_long(y) for y in years3],
            "--", color="#37474F", lw=2.2, alpha=0.5,
            label="Frontier training compute\n(growth slows ~1.4× yr⁻¹ after 2040)")

# scale_exp intuition for theory-heavy tasks:
#   ~0.42 → learning theory: some empirical work on large models, but mostly proof-based
#   ~0.18 → complexity: almost purely theoretical, minimal experimental validation
#   ~0.22 → info theory: some empirical; channel capacity experiments on models
#   ~0.32 → cryptography: post-quantum lattice computations scale modestly
tasks3 = [
    ("Learning Theory\n(generalisation bounds, PAC-Bayes,\nneural scaling law derivations)",
     3e19, 0.42, "#1565C0", "o"),
    ("Computational Complexity\n(hardness results, reductions,\nquery complexity of learning)",
     5e15, 0.18, "#00796B", "s"),
    ("Information Theory\n(capacity bounds, compression,\nKolmogorov complexity of models)",
     1e16, 0.22, "#6A1B9A", "^"),
    ("Cryptography\n(post-quantum, lattice-based,\nAI-robustness of crypto systems)",
     8e17, 0.32, "#BF360C", "D"),
]

for label, base, exp, color, marker in tasks3:
    vals = [task_raw(base, exp, y, frontier_fn=frontier_long) for y in years3]
    ax.semilogy(years3, vals, "-", color=color, lw=2.5, label=label,
                marker=marker, markevery=me3, ms=8,
                markeredgewidth=0.6, markeredgecolor="white")

# Slowdown reference line
ax.axvline(2040, color="#F57F17", lw=1.5, ls="--", alpha=0.65)
ax.text(2040.4, 8e14, "Assumed\nhardware\nceiling\n(2040)", color="#E65100",
        fontsize=8.5, va="bottom", alpha=0.85)

# AGI milestone
ax.axvline(2055, color="#B71C1C", lw=2.2, ls=":", alpha=0.9)
ax.text(2054.8, 3e35, "AGI ≈ 2055", color="#B71C1C", fontsize=10,
        fontweight="bold", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#B71C1C", alpha=0.9))
ax.axvspan(2050, 2055, alpha=0.055, color="#B71C1C", zorder=0,
           label="Critical compute window")

# Gap annotation at 2055
f2055 = frontier_long(2055)
cc_2055 = task_raw(5e15, 0.18, 2055, frontier_fn=frontier_long)
ax.annotate("", xy=(2055, f2055), xytext=(2055, cc_2055),
            arrowprops=dict(arrowstyle="<->", color="#607D8B", lw=1.3))
ax.text(2055.3, np.sqrt(f2055 * cc_2055),
        "Frontier ÷ complexity\ntheory ≈ 10¹⁶ gap\n(theoretical work remains\ntractable at any compute level)",
        fontsize=8, color="#546E7A", va="center")

ax.set_xlim(2024, 2057.5)
ax.set_ylim(5e12, 5e38)
ax.yaxis.set_major_formatter(fmt)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Compute per research campaign (FLOPs)", fontsize=12)
ax.set_title(
    "Scenario 3 — Long Timelines (AGI ≈ 2055)\n"
    "Compute Intensiveness of AI Safety Research Agendas",
    fontsize=13, fontweight="bold", pad=13,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.94, edgecolor="#ccc",
          labelspacing=0.6)
ax.text(0.99, 0.01,
        "Epoch AI effective compute framework | Frontier: ~2.5× yr⁻¹ (2024–2040), ~1.4× yr⁻¹ (2040+)\n"
        "Theory-heavy tasks (complexity, info theory) scale minimally; gap vs. frontier widens over decades",
        transform=ax.transAxes, fontsize=7.5, color="#888", ha="right", va="bottom")

plt.tight_layout()
plt.savefig("/Users/eleni/compute_models/scenario3_long_timelines.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scenario3_long_timelines.png")

print("\nAll 3 scenario graphs complete.")
