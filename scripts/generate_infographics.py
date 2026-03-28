"""Generate dark-theme infographics for OASIS Crisis Communication Optimizer."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Dark theme colors
BG   = "#0D1117"
TEXT = "#E6EDF3"
GRID = "#30363D"
PURPLE = "#7B61FF"
BLUE   = "#00C2FF"
GREEN  = "#00E5A0"
WARN   = "#FF9500"
RED    = "#FF4D4F"

ASSETS = Path(__file__).parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

def apply_dark(fig, ax_list):
    fig.patch.set_facecolor(BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.8)

# ── 1. Strategy comparison bar chart ──────────────────────────────────────────
print("Generating strategy_comparison.png …")
fig, ax = plt.subplots(figsize=(10, 6))
apply_dark(fig, ax)

strategies  = ["Fear-Based", "Empathy-Led", "Authority", "Community\nTrust", "Data-Driven"]
compliance  = [0.41, 0.73, 0.56, 0.68, 0.64]
trust_delta = [0.12, 0.38, 0.20, 0.42, 0.29]

x = np.arange(len(strategies))
w = 0.38
bars1 = ax.bar(x - w/2, compliance, w, color=PURPLE, alpha=0.9, label="Compliance Rate")
bars2 = ax.bar(x + w/2, trust_delta, w, color=GREEN, alpha=0.9, label="Trust Δ")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", color=TEXT, fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(strategies, color=TEXT, fontsize=10)
ax.set_ylabel("Score", color=TEXT)
ax.set_title("Messaging Strategy Comparison — Compliance & Trust", color=TEXT, fontsize=13, pad=12)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
ax.set_ylim(0, 0.85)
plt.tight_layout()
plt.savefig(ASSETS / "strategy_comparison.png", dpi=150, facecolor=BG)
plt.close()
print(f"  ✓  {ASSETS / 'strategy_comparison.png'}")

# ── 2. Simulation pipeline diagram ────────────────────────────────────────────
print("Generating pipeline_diagram.png …")
fig, ax = plt.subplots(figsize=(12, 5))
apply_dark(fig, ax)
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("OASIS Crisis Communication — Simulation Pipeline", color=TEXT, fontsize=13, pad=10)

boxes = [
    (0.4, 2.0, "Social\nNetwork\nGraph",    PURPLE),
    (2.4, 2.0, "Crisis\nEvent\nInjection",  BLUE),
    (4.4, 2.0, "Messaging\nStrategy\nA/B",  WARN),
    (6.4, 2.0, "Agent\nBelief\nUpdate",     GREEN),
    (8.4, 2.0, "Metric\nCollection",        PURPLE),
    (10.4, 2.0, "Report\n&\nRanking",       BLUE),
]
for (bx, by, label, color) in boxes:
    rect = mpatches.FancyBboxPatch((bx, by), 1.7, 1.2,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor=GRID, alpha=0.85)
    ax.add_patch(rect)
    ax.text(bx + 0.85, by + 0.6, label, ha="center", va="center",
            color=BG, fontsize=9, fontweight="bold")

for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i+1][0] + 0.05, boxes[i+1][1] + 0.6),
                xytext=(boxes[i][0] + 1.75, boxes[i][1] + 0.6),
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.5))

labels = ["500 LLM agents\nN≤1M scalable", "Outbreak /\ndisinfo event",
          "5 strategies\nrandomized", "Bayesian update\nper step", "Gini·trust·spread",
          "A/B winner +\nconfidence"]
for (bx, by, _, _c), lbl in zip(boxes, labels):
    ax.text(bx + 0.85, by - 0.3, lbl, ha="center", va="top",
            color=TEXT, fontsize=7.5, alpha=0.8)

plt.tight_layout()
plt.savefig(ASSETS / "pipeline_diagram.png", dpi=150, facecolor=BG)
plt.close()
print(f"  ✓  {ASSETS / 'pipeline_diagram.png'}")

# ── 3. Belief propagation over time ───────────────────────────────────────────
print("Generating belief_propagation.png …")
fig, ax = plt.subplots(figsize=(10, 6))
apply_dark(fig, ax)

steps = np.arange(0, 51)
np.random.seed(42)

def belief_curve(start, final, noise=0.015):
    t = steps / 50
    curve = start + (final - start) * (3*t**2 - 2*t**3)
    return curve + np.random.normal(0, noise, len(steps))

ax.plot(steps, belief_curve(0.35, 0.73), color=GREEN,  lw=2.5, label="Empathy-Led")
ax.plot(steps, belief_curve(0.35, 0.68), color=BLUE,   lw=2.5, label="Community Trust")
ax.plot(steps, belief_curve(0.35, 0.64), color=PURPLE, lw=2.0, label="Data-Driven")
ax.plot(steps, belief_curve(0.35, 0.56), color=WARN,   lw=2.0, label="Authority")
ax.plot(steps, belief_curve(0.35, 0.41), color=RED,    lw=1.5, label="Fear-Based", linestyle="--")

ax.axvline(10, color=TEXT, lw=1, alpha=0.5, linestyle=":")
ax.text(10.5, 0.37, "Crisis onset", color=TEXT, fontsize=9, alpha=0.8)
ax.set_xlabel("Simulation Steps", color=TEXT)
ax.set_ylabel("Population Belief Alignment", color=TEXT)
ax.set_title("Belief Propagation by Messaging Strategy", color=TEXT, fontsize=13, pad=12)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="lower right")
ax.set_ylim(0.25, 0.85)
plt.tight_layout()
plt.savefig(ASSETS / "belief_propagation.png", dpi=150, facecolor=BG)
plt.close()
print(f"  ✓  {ASSETS / 'belief_propagation.png'}")

# ── 4. Network topology impact radar ─────────────────────────────────────────
print("Generating network_impact_radar.png …")
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT)

categories = ["Compliance\nRate", "Trust\nGrowth", "Spread\nControl",
              "Misinformation\nResistance", "Community\nCohesion"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

empathy  = [0.73, 0.76, 0.68, 0.71, 0.80]
fear     = [0.41, 0.24, 0.35, 0.28, 0.32]
data     = [0.64, 0.58, 0.72, 0.65, 0.55]

for vals, color, label in [
    (empathy, GREEN, "Empathy-Led"),
    (fear, RED, "Fear-Based"),
    (data, BLUE, "Data-Driven"),
]:
    vals += vals[:1]
    ax.plot(angles, vals, color=color, lw=2, label=label)
    ax.fill(angles, vals, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color=TEXT, size=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color=TEXT, size=8)
ax.spines["polar"].set_color(GRID)
ax.grid(color=GRID, linewidth=0.7, alpha=0.7)
ax.set_title("Strategy Performance Radar", color=TEXT, fontsize=13, pad=20)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(ASSETS / "network_impact_radar.png", dpi=150, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  ✓  {ASSETS / 'network_impact_radar.png'}")

print(f"\nAll charts saved to: {ASSETS}/")
