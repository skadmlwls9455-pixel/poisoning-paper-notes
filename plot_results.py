"""results/results_seed*.json 을 읽어서 회복 곡선을 그림.

시드가 여러 개면 굵은 선 = 평균, 옅은 띠 = 시드 간 최소~최대.
"""
import glob, json
from collections import defaultdict
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]   # N이 작은 순서대로 고정
MODES = ["clean", "corrective"]

runs = []
for f in sorted(glob.glob("results/results_seed*.json")):
    runs += json.load(open(f))
seeds = sorted({r["seed"] for r in runs})
ns = sorted({r["n_poison"] for r in runs})

# (mode, n) -> step -> [seed별 ASR]
curves = defaultdict(lambda: defaultdict(list))
for r in runs:
    for step, a in r["curve"]:
        curves[r["mode"], r["n_poison"]][step].append(a)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
WASH_END = max(step for c in curves.values() for step in c)
for ax, mode in zip(axes, MODES):
    ends = {}
    for color, n in zip(COLORS, ns):
        c = curves[mode, n]
        if not c: continue
        xs = sorted(c)
        mean = [sum(c[x]) / len(c[x]) for x in xs]
        ax.fill_between(xs, [min(c[x]) for x in xs], [max(c[x]) for x in xs], color=color, alpha=.15, lw=0)
        ax.plot(xs, mean, color=color, lw=2, marker="o", ms=5, label=f"N={n}")
        ends.setdefault(round(mean[-1], 2), []).append(n)
    # 끝 값이 비슷한 선은 겹쳐 그려지니까 라벨을 하나로 묶음
    groups = []
    for y in sorted(ends):
        if groups and y - groups[-1][0] < 0.06: groups[-1][1].extend(ends[y])
        else: groups.append([y, list(ends[y])])
    for y, group in groups:
        ax.annotate("N=" + ", ".join(map(str, sorted(group))), (WASH_END, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#444")
    ax.set_title(f"wash: {mode}", loc="left")
    ax.set_xlabel("wash steps")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=.25)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_ylabel("attack success rate")
axes[-1].legend(frameon=False, loc="center right", bbox_to_anchor=(1.45, 0.5))
fig.suptitle(f"GPT-2 backdoor recovery curves (seeds: {', '.join(map(str, seeds))})", x=0.06, ha="left")
plt.tight_layout()
plt.savefig("results/recovery_curves.png", dpi=150, bbox_inches="tight")
print("저장: results/recovery_curves.png")
