"""오염 개수 N에 따른 오염 직후 ASR 곡선 + 로지스틱 맞춤.

results/threshold_seed*.json (오염 단계만 돌린 것)과 results/results_seed*.json (세척까지 돌린 것)의
오염 직후 ASR을 (시드, N)별로 모아서 그림. 같은 (시드, N)이 둘 다 있으면 하나만 씀(같은 계산이라).

맞추는 식: ASR(N) = 1 / (1 + exp(-k * (ln N - ln N50)))
  N50 = ASR이 50%가 되는 오염 개수, k = 곡선이 얼마나 가파른지
"""
import glob, json, sys
import numpy as np
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results"
pts = {}
for f in sorted(glob.glob(f"{d}/threshold_seed*.json")) + sorted(glob.glob(f"{d}/results_seed*.json")):
    for r in json.load(open(f)):
        pts.setdefault((r["seed"], r["n_poison"]), r["asr_before"])
seeds = sorted({s for s, _ in pts})
ns = sorted({n for _, n in pts})
x = np.array([n for _, n in pts], float)
y = np.array(list(pts.values()))


@np.errstate(over="ignore")
def logistic(n, n50, k):
    n = np.maximum(n, 1e-9)
    return 1 / (1 + np.exp(-k * (np.log(n) - np.log(n50))))


# 격자 탐색으로 최소제곱 맞춤 (scipy 없이)
best = None
for n50 in np.exp(np.linspace(np.log(1), np.log(2000), 600)):
    for k in np.exp(np.linspace(np.log(0.2), np.log(200), 300)):
        err = ((logistic(x, n50, k) - y) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, n50, k)
_, n50, k = best

# N50 불확실성: 시드를 복원추출로 다시 뽑아서 다시 맞춤 (시드가 2개 이상일 때만)
ci = None
if len(seeds) > 1:
    rng = np.random.default_rng(0); boots = []
    grid50 = np.exp(np.linspace(np.log(1), np.log(2000), 200)); gridk = np.exp(np.linspace(np.log(0.2), np.log(200), 100))
    for _ in range(200):
        pick = rng.choice(seeds, len(seeds))
        xb = np.concatenate([[n for (s, n) in pts if s == p] for p in pick]).astype(float)
        yb = np.concatenate([[v for (s, n), v in pts.items() if s == p] for p in pick])
        errs = [[((logistic(xb, a, b) - yb) ** 2).sum() for b in gridk] for a in grid50]
        boots.append(grid50[np.unravel_index(np.argmin(errs), (200, 100))[0]])
    ci = np.percentile(boots, [2.5, 97.5])

print(f"시드 {seeds}, N {ns}")
for n in ns:
    v = [pts[s, n] for s in seeds if (s, n) in pts]
    print(f"  N={n:4d}: ASR " + ", ".join(f"{a:.2f}" for a in v))
print(f"로지스틱 맞춤: N50 = {n50:.1f}" + (f"  (95% 부트스트랩 구간 {ci[0]:.1f}~{ci[1]:.1f})" if ci is not None else "")
      + f", 기울기 k = {k:.1f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
xp = np.where(x == 0, 0.6, x)   # 로그 축이라 N=0은 왼쪽 끝에 따로 표시
jit = np.array([(s - np.mean(seeds)) * 0.04 for s, _ in pts])
ax.scatter(xp * np.exp(jit), y, s=36, color="#2a78d6", alpha=.75, zorder=3, label="each seed")
xs = np.exp(np.linspace(0, np.log(max(ns) * 1.3), 300))
ax.plot(xs, logistic(xs, n50, k), color="#eb6834", lw=2, label=f"logistic fit (N50 = {n50:.0f})")
ax.axvline(n50, color="#eb6834", lw=1, ls=":")
ax.set_xscale("log")
ticks = [t for t in [10, 20, 30, 50, 100, 250] if t <= max(ns)]   # 로그 축에서 안 겹치는 것만
ax.set_xticks([0.6] + ticks, minor=False)
ax.set_xticklabels(["0"] + [str(t) for t in ticks])
ax.set_xticks([], minor=True)
ax.set_xlabel("poison samples N (clean = 2000)")
ax.set_ylabel("ASR right after poisoning")
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.25)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"GPT-2 backdoor threshold (seeds: {', '.join(map(str, seeds))})", loc="left")
plt.tight_layout()
plt.savefig(f"{d}/threshold_curve.png", dpi=150, bbox_inches="tight")
print(f"저장: {d}/threshold_curve.png")
