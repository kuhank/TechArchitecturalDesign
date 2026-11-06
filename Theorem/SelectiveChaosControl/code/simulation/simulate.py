"""Numerical study for the Selective Chaos Control (SCC) framework.

Experiments (all synthetic, fully reproducible, seed-controlled):

  E1  Dimension-wise divergence in canonical chaotic flows (Rossler, Lorenz):
      per-coordinate separation of twin trajectories; dimension-wise FTLE.
  E2  Divergence-error bracket: empirical check of
      1 - sqrt(1-d) <= p <= (1 - sqrt(1-2d))/2.
  E3  Synthetic structured-output agent (D semantic dimensions):
      single / whole-output self-consistency / verify-all / factorised
      consensus / SCC.  Accuracy vs. cost, threshold sweep (Pareto front),
      localisation AUROC of the per-dimension exponent.
  E4  Multi-step agents with heterogeneous error growth:
      fixed checkpointing vs. Lyapunov-time (adaptive) checkpointing.
  E5  Selective chaos: per-dimension temperature feedback vs. global temperature.

Outputs: figures (PDF/PNG) in ../../paper/figures and results.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C = {"single": "#7f7f7f", "selfcons": "#1f77b4", "verifyall": "#9467bd",
     "factor": "#2ca02c", "scc": "#d62728", "fixed": "#1f77b4", "adaptive": "#d62728"}
RESULTS: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------- E1
def rossler(t, s, a=0.2, b=0.2, c=5.7):
    x, y, z = s
    return [-y - z, x + a * y, b + z * (x - c)]


def lorenz(t, s, sig=10.0, r=28.0, b=8 / 3):
    x, y, z = s
    return [sig * (y - x), x * (r - z) - y, x * y - b * z]


def twin_divergence(f, s0, T, eps=1e-8, n=4000, burn=100.0):
    s0 = solve_ivp(f, (0, burn), s0, rtol=1e-11, atol=1e-11).y[:, -1]
    t = np.linspace(0, T, n)
    a = solve_ivp(f, (0, T), s0, t_eval=t, rtol=1e-11, atol=1e-11).y
    b = solve_ivp(f, (0, T), s0 + np.array([eps, 0, 0]), t_eval=t, rtol=1e-11, atol=1e-11).y
    return t, a, np.abs(a - b), eps


def e1():
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    keep = {}
    for name, f, s0, T, ax in [("Rossler", rossler, [1, 1, 1], 350.0, axes[0]),
                               ("Lorenz", lorenz, [1, 1, 1], 30.0, axes[1])]:
        t, a, d, eps = twin_divergence(f, np.array(s0, float), T)
        span = a.max(axis=1) - a.min(axis=1)           # attractor extent per coordinate
        dn = d / span[:, None]                          # normalised per-dimension divergence
        keep[name] = (t, dn)
        lam = {}
        for j, lab in enumerate("xyz"):
            ax.semilogy(t, np.maximum(dn[j], 1e-14), lw=0.6, label=f"${lab}$")
            # dimension-wise FTLE: slope of ln d_j over the exponential-growth window
            m = (dn[j] > 1e-11) & (dn[j] < 1e-2)
            lam[lab] = float(np.polyfit(t[m], np.log(dn[j][m]), 1)[0]) if m.sum() > 10 else float("nan")
            hit = np.argmax(dn[j] > 0.1)
            lam[lab + "_T10"] = float(t[hit]) if hit > 0 else float("nan")
        ax.axhline(0.1, color="k", ls=":", lw=0.6)
        ax.set_title(f"({'a' if name == 'Rossler' else 'b'}) {name}")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$d_j(t)$ / attractor extent")
        ax.legend(loc="lower right", frameon=False, ncol=3, handlelength=1.2)
        out[name] = lam

    # finite-time disagreement between dimensions: log10(d_z / d_x), d_y / d_x
    t, dn = keep["Rossler"]
    w = 40
    for j, lab, col in [(2, "z", "#2ca02c"), (1, "y", "#ff7f0e")]:
        r = np.log10(np.maximum(dn[j], 1e-16) / np.maximum(dn[0], 1e-16))
        rm = np.convolve(r, np.ones(w) / w, mode="same")
        axes[2].plot(t, r, lw=0.3, color=col, alpha=0.35)
        axes[2].plot(t, rm, lw=1.0, color=col, label=rf"$\log_{{10}}(d_{lab}/d_x)$")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_title("(c) Rossler: dimension ratio")
    axes[2].set_xlabel("time")
    axes[2].set_ylabel("orders of magnitude")
    axes[2].legend(frameon=False, loc="lower left")
    rz = np.log10(np.maximum(dn[2], 1e-16) / np.maximum(dn[0], 1e-16))
    out["Rossler_ratio_z_over_x_log10_p5_p95"] = [float(np.percentile(rz, 5)), float(np.percentile(rz, 95))]
    save(fig, "fig1_dimensionwise_divergence")
    RESULTS["E1"] = out


# ----------------------------------------------------------------- E2
def e2(rng):
    from scc.sensor import implied_error_rate, implied_error_rate_lower, pairwise_divergence
    ps = np.linspace(0.01, 0.45, 30)
    rows = []
    for p in ps:
        for m, lab in [(1, "single wrong alternative"), (50, "diffuse (50 alternatives)")]:
            ds = []
            for _ in range(400):
                k = 7
                vals = ["ok" if rng.random() > p else f"w{rng.integers(m)}" for _ in range(k)]
                ds.append(pairwise_divergence(vals))
            rows.append((p, m, float(np.mean(ds))))
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    dgrid = np.linspace(0, 0.5, 200)
    ax.fill_between(dgrid, [implied_error_rate_lower(d) for d in dgrid],
                    [implied_error_rate(d) for d in dgrid], color="0.85", label="theoretical bracket")
    dl = np.linspace(0, 0.65, 200)
    ax.plot(dl, [implied_error_rate_lower(d) for d in dl], color="0.5", lw=0.7, ls="--",
            label=r"lower bound $1-\sqrt{1-d}$")
    for m, mk, lab in [(1, "o", "single wrong alt."), (50, "s", "diffuse wrong ans.")]:
        r = rows[rows[:, 1] == m]
        ax.plot(r[:, 2], r[:, 0], mk, ms=3, label=lab)
    ax.set_xlabel(r"observed divergence $d_j$")
    ax.set_ylabel(r"true error rate $p_j$")
    ax.set_xlim(0, 0.62)
    ax.legend(frameon=False)
    save(fig, "fig2_divergence_error_bracket")
    dom = rows[rows[:, 2] <= 0.5]
    near = strict = 0
    for p, m, d in dom:
        lo, hi = implied_error_rate_lower(d), implied_error_rate(d)
        near += (lo - 0.03) <= p <= (hi + 0.03)
        strict += lo <= p <= hi
    RESULTS["E2"] = {"points": int(len(rows)), "points_in_domain_d<=0.5": int(len(dom)),
                     "in_domain_within_0.03": int(near), "in_domain_strictly_inside": int(strict)}


# ----------------------------------------------------------------- E3
D = 6
WEIGHTS = np.ones(D)


def make_queries(rng, n):
    """Latent per-dimension error rate at temperature 0.
    Most dimensions are easy; a few are hard (heavy tail) - mirrors the
    observation that LLM errors concentrate in one component of the output."""
    base = rng.beta(0.6, 12.0, size=(n, D))                 # mostly ~0.0-0.1
    hard = rng.random((n, D)) < 0.12                        # ~12 % of dims are hard
    base[hard] = rng.uniform(0.25, 0.65, size=hard.sum())
    return np.clip(base, 0.0, 0.9)


def sample_dims(rng, p, k, m=8, kappa=0.15, tau=0.7):
    """k samples; each dim correct (0) w.p. 1-p(tau) else one of m wrong values."""
    pt = np.clip(p + kappa * tau * (p > 0.02), 0, 0.95)
    wrong = rng.random((k, D)) < pt
    val = np.where(wrong, 1 + rng.integers(0, m, size=(k, D)), 0)
    return val


TIE_RNG = np.random.default_rng(11)


def plurality(col, rng=None):
    """Plurality value with ties broken uniformly at random."""
    rng = rng or TIE_RNG
    vals, cnt = np.unique(col, return_counts=True)
    top = np.flatnonzero(cnt == cnt.max())
    return vals[rng.choice(top)], cnt.max() / len(col)


def run_strategies(rng, P, k=5, theta=0.7, g=0.25):
    from scc.sensor import pairwise_divergence
    n = len(P)
    acc = {s: 0 for s in ["single", "selfcons", "verifyall", "factor", "scc"]}
    cost = {s: 0.0 for s in acc}
    dim_err, dim_lam = [], []
    eps = 0.15
    for p in P:
        # single greedy (tau=0)
        greedy = np.where(rng.random(D) < p, 1 + rng.integers(0, 8, D), 0)
        acc["single"] += np.all(greedy == 0)
        cost["single"] += 1

        S = sample_dims(rng, p, k)
        # whole-output self-consistency: plurality over complete tuples
        tuples = [tuple(r) for r in S]
        uniq, cnt = np.unique(np.array(tuples), axis=0, return_counts=True)
        top = np.flatnonzero(cnt == cnt.max())
        best = uniq[rng.choice(top)]                       # ties broken at random
        acc["selfcons"] += np.all(best == 0)
        cost["selfcons"] += k

        # verify-all: one generation, then ground every dimension
        va = np.where(rng.random(D) < p * g, 1, 0)
        acc["verifyall"] += np.all(va == 0)
        cost["verifyall"] += 1 + D

        # factorised consensus (ablation: SCC without grounding)
        fc = np.array([plurality(S[:, j], rng)[0] for j in range(D)])
        acc["factor"] += np.all(fc == 0)
        cost["factor"] += k

        # SCC: per-dimension divergence -> consensus or grounding
        out = np.zeros(D, int)
        extra = 0
        for j in range(D):
            d = pairwise_divergence(list(S[:, j]))
            lam = math.log(max(d, 1e-3) / eps)
            dim_lam.append(lam)
            v, share = plurality(S[:, j], rng)
            dim_err.append(int(v != 0))
            if d <= theta:
                out[j] = v
            else:
                extra += 1
                out[j] = 0 if rng.random() > p[j] * g else 1
        acc["scc"] += np.all(out == 0)
        cost["scc"] += k + extra
    return ({s: acc[s] / n for s in acc}, {s: cost[s] / n for s in cost},
            np.array(dim_err), np.array(dim_lam))


def auroc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def e3(rng):
    P = make_queries(rng, 4000)
    acc, cost, err, lam = run_strategies(rng, P)
    au = auroc(err, lam)

    # threshold sweep for SCC (Pareto front)
    sweep = []
    for th in [0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]:
        a, c, _, _ = run_strategies(np.random.default_rng(7), P[:2000], theta=th)
        sweep.append((th, a["scc"], c["scc"]))
    # sensitivity to grounding strength g
    gsweep = []
    for g in [0.05, 0.10, 0.25, 0.50]:
        a, c, _, _ = run_strategies(np.random.default_rng(7), P[:2000], g=g)
        gsweep.append((g, a["verifyall"], c["verifyall"], a["scc"], c["scc"]))
    # self-consistency with growing K
    sc = []
    for k in [1, 3, 5, 9, 15]:
        a, c, _, _ = run_strategies(np.random.default_rng(7), P[:2000], k=k)
        sc.append((k, a["selfcons"], c["selfcons"], a["factor"], c["factor"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    names = {"single": "Single", "selfcons": "Self-cons. (whole)", "verifyall": "Verify-all",
             "factor": "Factorised cons.", "scc": "SCC (proposed)"}
    for s in names:
        ax.scatter(cost[s], acc[s], color=C[s], s=28, zorder=3, label=names[s])
    sw = np.array(sweep)
    ax.plot(sw[:, 2], sw[:, 1], "-", color=C["scc"], lw=0.8, alpha=0.6, label=r"SCC, $\theta$ sweep")
    scn = np.array(sc)
    ax.plot(scn[:, 2], scn[:, 1], "--", color=C["selfcons"], lw=0.8, alpha=0.6, label="Self-cons., K sweep")
    ax.plot(scn[:, 4], scn[:, 3], ":", color=C["factor"], lw=0.8, alpha=0.8, label="Factorised, K sweep")
    ax.set_xlabel("model calls per query")
    ax.set_ylabel("full-output accuracy")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5, ncol=2)
    ax.set_title("(a) Accuracy vs. cost")

    ax = axes[1]
    dvals = np.exp(lam) * 0.15
    levels = np.round(dvals, 2)
    uniq = sorted(set(levels.tolist()))
    rates = [err[levels == u].mean() for u in uniq]
    counts = [(levels == u).sum() for u in uniq]
    xs = np.arange(len(uniq))
    ax.bar(xs, rates, color=C["scc"], alpha=0.8)
    for x, c in zip(xs, counts):
        ax.text(x, 0.02, f"n={c}", rotation=90, ha="center", va="bottom", fontsize=6, color="w")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{u:.2f}" if u > 0.0015 else "0" for u in uniq], fontsize=7)
    ax.set_xlabel(r"observed dimension divergence $d_j$ ($K=5$)")
    ax.set_ylabel("P(consensus value wrong)")
    ax.set_title(f"(b) Localisation: AUROC of $\\hat\\lambda_j$ = {au:.3f}")
    RESULTS_LOCAL = {"levels": uniq, "p_wrong": [float(r) for r in rates], "n": [int(c) for c in counts]}
    save(fig, "fig3_agent_accuracy_cost")
    RESULTS["E3"] = {"wrong_rate_by_divergence": RESULTS_LOCAL, "accuracy": acc, "calls": cost, "auroc_dim_exponent": au,
                     "scc_threshold_sweep": sweep, "grounding_strength_sweep_(g,va_acc,va_calls,scc_acc,scc_calls)": gsweep,
                     "selfcons_vs_k": sc}


# ----------------------------------------------------------------- E4
def run_agents(rng, lams, L=40, e0=0.01, delta=0.3, policy="fixed", interval=5, alpha=0.7,
               noise=0.15, obs_noise=0.3, first=2, thr=0.3, z=1.0):
    """Multi-step agents with deviation e_{t+1} = e_t exp(lambda + noise*xi).
    The true deviation is NOT observable between checkpoints. A checkpoint
    (verification) reveals a noisy reading e_obs = e_t exp(obs_noise*xi) and
    re-anchors e to e0. Failure = e_t > delta at any step (unnoticed).
    Policies:
      fixed    - checkpoint every `interval` steps
      known    - interval floor(alpha*T_h) with the true lambda (tests Prop. 5)
      ema      - lambda estimated online from checkpoint readings, exponential average
      ucb      - lambda estimated online, pooled over checkpoints, plus z standard errors
      oracle   - observes e_t at every step and re-anchors when e_t > thr*delta
                 (unattainable reference)"""
    lnr = math.log(delta / e0)
    succ, ckpts = 0, 0
    for lam in lams:
        e, since, ok = e0, 0, True
        nxt = max(1, int(alpha * lnr / lam)) if policy == "known" else first
        lam_hat, reads = None, []
        for t in range(L):
            e = e * math.exp(lam + noise * rng.standard_normal())
            since += 1
            if e > delta:
                ok = False
                break
            if policy == "fixed":
                trig = since >= interval
            elif policy == "none":
                trig = False
            elif policy == "oracle":
                trig = e > thr * delta
            else:
                trig = since >= nxt
            if trig:
                ckpts += 1
                if policy in ("ema", "ucb"):
                    e_obs = e * math.exp(obs_noise * rng.standard_normal())
                    lam_obs = math.log(max(e_obs, 1e-12) / e0) / since
                    if policy == "ema":
                        lam_hat = lam_obs if lam_hat is None else 0.5 * lam_hat + 0.5 * lam_obs
                        est = lam_hat
                    else:
                        reads.append((lam_obs, since))
                        w = np.array([n for _, n in reads], float)
                        v = np.array([l for l, _ in reads])
                        lam_hat = float((w * v).sum() / w.sum())
                        se = math.sqrt(noise ** 2 / w.sum() + obs_noise ** 2 * len(w) / w.sum() ** 2)
                        est = lam_hat + z * se
                    nxt = max(1, int(alpha * lnr / max(est, 1e-3)))
                e, since = e0, 0
        succ += ok
    return succ / len(lams), ckpts / len(lams)


def e4(rng):
    lams = rng.uniform(0.03, 0.7, 3000)
    R = lambda **kw: run_agents(np.random.default_rng(3), lams, **kw)  # noqa: E731
    fixed = [(N, *R(policy="fixed", interval=N)) for N in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40]]
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    known = [(a, *R(policy="known", alpha=a)) for a in alphas]
    ucb = [(a, *R(policy="ucb", alpha=a)) for a in alphas]
    ema = [(a, *R(policy="ema", alpha=a)) for a in alphas]
    oracle = [(th, *R(policy="oracle", thr=th)) for th in [0.1, 0.2, 0.3, 0.5, 0.7]]
    none = R(policy="none")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for data, mk, col, lab in [(fixed, "o-", C["fixed"], "fixed interval"),
                               (known, "s-", C["adaptive"], r"horizon, true $\lambda$"),
                               (ucb, "d--", "#ff7f0e", r"horizon, online $\hat\lambda$ (UCB)"),
                               (ema, "x:", "#8c564b", r"horizon, online $\hat\lambda$ (EMA)"),
                               (oracle, "^:", "0.4", r"oracle threshold (sees $e_t$)")]:
        d = np.array(data)
        ax.plot(d[:, 2], d[:, 1], mk, ms=3, lw=0.9, color=col, label=lab)
    ax.axhline(none[0], color="0.6", ls=":", lw=0.7)
    ax.set_xlim(0, 16)
    ax.set_xlabel("checkpoints per task")
    ax.set_ylabel("task success rate")
    ax.set_title("(a) Multi-step agents (L = 40)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)

    ax = axes[1]
    e0, delta = 0.01, 0.3
    al = np.linspace(0.05, 1.0, 100)
    ax.plot(al, e0 * (delta / e0) ** al, color="k")
    ax.axhline(delta, color=C["scc"], ls="--", lw=0.8, label=r"tolerance $\Delta$")
    ax.set_xlabel(r"checkpoint fraction $\alpha$ of horizon $T_h$")
    ax.set_ylabel(r"worst-case deviation $e_{\max}$")
    ax.set_title(r"(b) Bound $e_{\max}=e_0(\Delta/e_0)^{\alpha}$ (noise-free)")
    ax.legend(frameon=False)
    save(fig, "fig4_lyapunov_time_checkpointing")

    fx = np.array(sorted(fixed, key=lambda r: r[2]))
    def vs_fixed(rows):
        return [(float(a), float(c), float(s), float(np.interp(c, fx[:, 2], fx[:, 1]))) for a, s, c in rows]
    RESULTS["E4"] = {"fixed_(N,succ,ckpts)": fixed, "none": none, "oracle_(thr,succ,ckpts)": oracle,
                     "known_vs_fixed_(alpha,ckpts,succ,fixed_interp)": vs_fixed(known),
                     "ucb_vs_fixed_(alpha,ckpts,succ,fixed_interp)": vs_fixed(ucb),
                     "ema_vs_fixed_(alpha,ckpts,succ,fixed_interp)": vs_fixed(ema),
                     "lambda_mean_over_max": float(lams.mean() / lams.max())}


# ----------------------------------------------------------------- E5
def e5(rng):
    from scc.controller import DimensionController
    from scc.decompose import Dimension
    steps = 40
    dims = [Dimension("predicates"), Dimension("tables"), Dimension("explanation", role="creative")]
    a = {"predicates": 1.6, "tables": 0.8, "explanation": 1.0}   # sensitivity of d to tau
    ctrls = {d.name: DimensionController(d, tau=0.9, eta=2.0) for d in dims}
    hist = {n: [] for n in ctrls}
    dhist = {n: [] for n in ctrls}
    for _ in range(steps):
        for n, c in ctrls.items():
            d = (1 - math.exp(-a[n] * c.tau)) * 0.8 + 0.03 * rng.standard_normal()
            d = min(max(d, 0), 1)
            c.update_temperature(d)
            hist[n].append(c.tau)
            dhist[n].append(d)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for n in ctrls:
        axes[0].plot(hist[n], label=n)
        axes[1].plot(dhist[n], label=n)
    axes[0].set_xlabel("control step")
    axes[0].set_ylabel(r"temperature $\tau_j$")
    axes[0].set_title("(a) Per-dimension temperature")
    axes[0].legend(frameon=False, loc="center right")
    axes[1].axhspan(0.2, 0.6, color="0.9", label="creative band")
    axes[1].set_xlabel("control step")
    axes[1].set_ylabel(r"divergence $d_j$")
    axes[1].set_title("(b) Divergence per dimension")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.98), ncol=2)
    axes[1].set_ylim(0, 0.8)
    save(fig, "fig5_selective_chaos")

    # global temperature trade-off: one tau for all dims
    taus = np.linspace(0, 1.2, 50)
    prec_d = (1 - np.exp(-1.6 * taus)) * 0.8
    crea_d = (1 - np.exp(-1.0 * taus)) * 0.8
    feasible = taus[(prec_d <= 0.05) & (crea_d >= 0.2)]
    RESULTS["E5"] = {"final_tau": {n: hist[n][-1] for n in hist},
                     "final_d": {n: float(np.mean(dhist[n][-10:])) for n in dhist},
                     "global_tau_feasible_count": int(len(feasible))}


# ----------------------------------------------------------------- E6
def e6():
    """The temperature feedback law is itself a 1-D map:
        u_{n+1} = u_n - eta * (g(exp(u_n)) - d*),  u = ln tau,  g(tau) = c(1 - exp(-a tau)).
    Linear stability of the fixed point: |1 - eta * tau* g'(tau*)| < 1."""
    a, c, dstar = 1.0, 0.8, 0.4
    tau_star = -math.log(1 - dstar / c) / a
    slope = tau_star * c * a * math.exp(-a * tau_star)        # tau* g'(tau*)
    eta_crit = 2.0 / slope
    etas = np.linspace(0.05, 3.2 * eta_crit, 1400)
    pts_x, pts_y, lyap = [], [], []
    for eta in etas:
        u = math.log(0.9)
        acc, n = 0.0, 0
        for i in range(1500):
            tau = math.exp(min(u, 5.0))
            g = c * (1 - math.exp(-a * tau))
            dg = c * a * math.exp(-a * tau) * tau             # d g / d u
            if i >= 500:
                acc += math.log(abs(1 - eta * dg) + 1e-300)
                n += 1
            u = u - eta * (g - dstar)
            u = max(min(u, 5.0), -12.0)
            if i >= 1400:
                pts_x.append(eta)
                pts_y.append(math.exp(u))
        lyap.append(acc / n)
    lyap = np.array(lyap)

    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.0), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(np.array(pts_x) / eta_crit, pts_y, ",", color="k", alpha=0.5)
    axes[0].axvline(1.0, color=C["scc"], ls="--", lw=0.8)
    axes[0].set_ylabel(r"temperature $\tau_n$ (attractor)")
    axes[0].set_yscale("log")
    axes[0].set_title("Period-doubling of the temperature controller")
    axes[1].plot(etas / eta_crit, lyap, color="k", lw=0.7)
    axes[1].axhline(0, color="0.5", lw=0.6)
    axes[1].axvline(1.0, color=C["scc"], ls="--", lw=0.8)
    axes[1].set_xlabel(r"normalised gain $\eta/\eta_c$,  $\eta_c = 2/(\tau^* g'(\tau^*))$")
    axes[1].set_ylabel(r"$\lambda_{\mathrm{ctrl}}$")
    save(fig, "fig6_controller_bifurcation")
    # observed bifurcation: first gain where the attractor spreads over > 1 %
    X, Y = np.array(pts_x), np.array(pts_y)
    spread = np.array([np.ptp(Y[X == e]) / tau_star for e in etas])
    first_split = float(etas[np.argmax(spread > 0.01)] / eta_crit)
    hi = etas[-1]
    RESULTS["E6"] = {"tau_star": tau_star, "eta_crit": eta_crit,
                     "observed_bifurcation_eta_over_etac": first_split,
                     "max_ctrl_lyap": float(lyap.max()),
                     "period2_range_at_max_gain": [float(Y[X == hi].min()), float(Y[X == hi].max())]}


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    e1()
    e2(rng)
    e3(rng)
    e4(rng)
    e5(rng)
    e6()
    json.dump(RESULTS, open(Path(__file__).with_name("results.json"), "w"), indent=2, default=float)
    print(json.dumps(RESULTS, indent=1, default=float))
