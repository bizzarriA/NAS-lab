"""Helper functions for Lab 1 (NAS on HW-NAS-Bench).

Students do not need to read this file: everything here is "plumbing"
(data loading, plotting, the benchmark oracle). The interesting parts of
the lab (search algorithms) live in the notebook.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# -----------------------------------------------------------------------------
# Search space (NAS-Bench-201 cell)
# -----------------------------------------------------------------------------
OPS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
OP_SHORT = {"none": "zero", "skip_connect": "skip", "nor_conv_1x1": "c1x1",
            "nor_conv_3x3": "c3x3", "avg_pool_3x3": "pool"}
EDGES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]   # (destination, source)
OP_COLS = [f"op_e{dst}_{src}" for dst, src in EDGES]       # CSV columns, same order

DATASETS = ["cifar10", "cifar100", "imagenet16"]
DEVICES = ["edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga"]


def parse_arch_str(arch_str):
    """'|op~0|+|op~0|op~1|+|op~0|op~1|op~2|'  ->  tuple of 6 operations."""
    return tuple(tok.split("~")[0]
                 for node in arch_str.split("+") for tok in node.strip("|").split("|"))


def to_arch_str(ops):
    """Tuple of 6 operations  ->  NAS-Bench-201 string."""
    o = ops
    return f"|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|"


def _as_ops(arch):
    """Accepts a DataFrame row, an arch string or a tuple/list of ops."""
    if isinstance(arch, pd.Series):
        return tuple(arch[c] for c in OP_COLS) if OP_COLS[0] in arch else parse_arch_str(arch["arch_str"])
    if isinstance(arch, str):
        return parse_arch_str(arch)
    return tuple(arch)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_benchmark(path, dataset="cifar10", verbose=True):
    """Load the CSV (local path or URL) and return a tidy DataFrame.

    One row per architecture, with standard column names:
      arch_str, op_e1_0 ... op_e3_2, ops (tuple of 6 ops),
      acc (validation accuracy, used during the search), test_acc, cost (training time)
    """
    assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
    raw = pd.read_csv(path)

    def col(name):
        c = f"{dataset}_{name}"
        return raw[c].astype(float) if c in raw.columns else pd.Series(np.nan, index=raw.index)

    df = raw[["arch_str"] + OP_COLS].copy()
    df["ops"] = [tuple(r) for r in df[OP_COLS].itertuples(index=False)]
    df["acc"] = col("valid_acc")
    df["test_acc"] = col("test_acc")
    df["cost"] = col("train_time")
    df = df.dropna(subset=["acc"]).drop_duplicates("arch_str").reset_index(drop=True)

    if verbose:
        print(f"Loaded {len(df)} architectures | dataset = {dataset} ")
    return df


def load_HW_benchmark(path, dataset="cifar10", device="edgegpu", verbose=True):
    """Load the CSV (local path or URL) and return a tidy DataFrame.

    One row per architecture, with standard column names:
      arch_str, op_e1_0 ... op_e3_2, ops (tuple of 6 ops),
      acc (validation accuracy, used during the search), test_acc,
      lat, energy (on `device`), cost (training time), flops, params
    """
    assert dataset in DATASETS, f"dataset must be one of {DATASETS}"
    assert device in DEVICES, f"device must be one of {DEVICES}"
    raw = pd.read_csv(path)

    def col(name):
        c = f"{dataset}_{name}"
        return raw[c].astype(float) if c in raw.columns else pd.Series(np.nan, index=raw.index)

    df = raw[["arch_str"] + OP_COLS].copy()
    df["ops"] = [tuple(r) for r in df[OP_COLS].itertuples(index=False)]
    df["acc"] = col("valid_acc")
    df["test_acc"] = col("test_acc")
    df["lat"] = col(f"{device}_latency")
    df["energy"] = col(f"{device}_energy")
    df["cost"] = col("train_time")
    df["flops"] = col("flops")
    df["params"] = col("params")
    df = df.dropna(subset=["acc"]).drop_duplicates("arch_str").reset_index(drop=True)

    if verbose:
        print(f"Loaded {len(df)} architectures | dataset = {dataset} | device = {device}")
        missing = [c for c in ["lat", "energy", "cost"] if df[c].isna().all()]
        if missing:
            print(f"  note: {missing} not available for this dataset/device (filled with NaN)")
    return df


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_cell(arch, ax=None, title=None, show=True):
    """Draw a NB201 cell. `arch` can be a DataFrame row, an arch string or a tuple of ops."""
    ops = _as_ops(arch)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    color = {"none": "lightgray", "skip_connect": "tab:green", "nor_conv_1x1": "tab:orange",
             "nor_conv_3x3": "tab:red", "avg_pool_3x3": "tab:blue"}
    for (dst, src), op in zip(EDGES, ops):
        span = dst - src
        rad = 0.0 if span == 1 else (-0.45 if span == 2 else 0.55)
        ax.annotate("", xy=(dst, 0), xytext=(src, 0),
                    arrowprops=dict(arrowstyle="->", color=color[op], lw=2,
                                    ls=":" if op == "none" else "-",
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=14, shrinkB=14))
        ylab = 0.08 if span == 1 else -rad * span / 2 + (0.07 if rad < 0 else -0.12)
        ax.text((src + dst) / 2, ylab, OP_SHORT[op], ha="center", fontsize=9, color=color[op])
    for n in range(4):
        ax.add_patch(plt.Circle((n, 0), 0.13, color="white", ec="black", zorder=3))
        ax.text(n, 0, ["in", "1", "2", "out"][n], ha="center", va="center", zorder=4)
    ax.set_xlim(-0.4, 3.4); ax.set_ylim(-1.0, 0.7); ax.axis("off")
    ax.set_title(title or to_arch_str(ops), fontsize=10)
    if show:
        plt.show()
    return ax


# -----------------------------------------------------------------------------
# Performance estimation strategy: the benchmark "oracle"
# -----------------------------------------------------------------------------
class BudgetExhausted(Exception):
    pass

def base_objective_function(arch):
    """Default objective: the validation accuracy of an architecture (a row of df)."""
    return arch["acc"]


class Oracle:
    """Tabular-benchmark estimation strategy with a query budget.

    oracle.query(idx)       -> objective of the architecture in row `idx`
    oracle.query_ops(ops)   -> same, given a tuple of 6 ops (or an arch string)
    oracle.remaining()      -> queries still available
    oracle.n_queries        -> queries used so far
    oracle.best()           -> (ops, objective) of the best architecture found
    oracle.incumbent_curve()-> best objective found after each query
    oracle.queried()        -> row indices of the queried architectures, in query order
    oracle.incumbent_idx()  -> row index of the best architecture after each query
    oracle.sim_cost         -> simulated training time spent (seconds)

    The objective is `objective_fn(row)`, where `row` is a row of `df`
    (e.g. row["acc"], row["lat"], row["energy"]). By default it is row[metric].
    Re-querying an architecture is free (results are cached).
    With noise_std > 0 the returned value is noisy (low-fidelity estimate);
    history and curves always record the TRUE value.
    """

    def __init__(self, df, budget, metric="acc", noise_std=0.0, seed=0, objective_fn=None):
        self.df, self.budget = df, budget
        self.costs = df["cost"].to_numpy() if "cost" in df else np.zeros(len(df))
        self.arch2idx = {ops: i for i, ops in enumerate(df["ops"])}
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.history = []          # list of (idx, true value)
        self.cache = {}
        self.sim_cost = 0.0
        self.objective_fn = objective_fn if objective_fn is not None else (lambda row: row[metric])

    @property
    def n_queries(self):
        return len(self.history)

    def remaining(self):
        return self.budget - self.n_queries

    def query(self, idx):
        idx = int(idx)
        if idx in self.cache:
            return self.cache[idx]
        if self.remaining() <= 0:
            raise BudgetExhausted()
        true = float(self.objective_fn(self.df.iloc[idx]))
        obs = true + (self.rng.normal(0, self.noise_std) if self.noise_std > 0 else 0.0)
        self.history.append((idx, true))
        self.cache[idx] = obs
        c = self.costs[idx]
        self.sim_cost += 0.0 if np.isnan(c) else c
        return obs

    def query_ops(self, ops):
        return self.query(self.arch2idx[_as_ops(ops)])

    def incumbent_curve(self):
        vals = np.array([v for _, v in self.history])
        return np.maximum.accumulate(vals) if len(vals) else vals

    def queried(self):
        return np.array([i for i, _ in self.history], dtype=int)

    def incumbent_idx(self):
        best_i, best_v, out = None, -np.inf, []
        for i, v in self.history:
            if v > best_v:
                best_i, best_v = i, v
            out.append(best_i)
        return np.array(out, dtype=int)

    def best(self):
        i, v = max(self.history, key=lambda t: t[1])
        return self.df["ops"].iat[i], float(v)


def evaluate_objective(df, objective_fn, idx=None):
    """Objective of every architecture of df (or only of the rows `idx`), as a numpy array."""
    sub = df if idx is None else df.iloc[idx]
    return np.array([objective_fn(row) for _, row in sub.iterrows()], dtype=float)

class LatencyCheckOracle(Oracle):
    # Oracle whose latency constraint is checked after the architecture has been picked

    def __init__(self, df, budget, lat_max, seed=0):
        super().__init__(df, budget, seed=seed)
        self.lat = df["lat"].to_numpy()
        self.lat_max = lat_max
        self.measured_lat = {}
        self.n_rejected = 0

    def query(self, idx):
        idx = int(idx)
        if idx in self.cache:
            return self.cache[idx]
        if self.remaining() <= 0:
            raise BudgetExhausted()
        self.measured_lat[idx] = self.lat[idx]
        if self.lat[idx] > self.lat_max:
            # Measured but not trained: costs one query, no training time
            self.history.append((idx, 0.0))
            self.cache[idx] = None
            self.n_rejected += 1
            return None
        return super().query(idx)

# -----------------------------------------------------------------------------
# Bayesian optimization: surrogate model
# -----------------------------------------------------------------------------
def make_gp(seed=0):
    """Gaussian Process with Matern kernel + noise, used as BO surrogate."""
    kernel = (ConstantKernel(1.0, (1e-2, 1e3))
              * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2))
              + WhiteKernel(1e-2, (1e-6, 1e1)))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=0, random_state=seed)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_regret(results, title="Regret vs number of evaluations", ylabel="Regret (% points)"):
    """results: dict name -> array (n_runs x budget) of regret curves
    (or a (curves, oracles) tuple, as returned by run_many)."""
    plt.figure(figsize=(9, 5))
    for name, out in results.items():
        R, _ = out if isinstance(out, tuple) else (out, None)
        R = np.atleast_2d(R)
        x = np.arange(1, R.shape[1] + 1)
        m = R.mean(0)
        plt.plot(x, m, label=name, lw=2)
        if len(R) > 1:
            se = R.std(0) / np.sqrt(len(R))
            plt.fill_between(x, m - se, m + se, alpha=0.2)
    plt.yscale("symlog", linthresh=0.1)
    plt.xlabel("# evaluated architectures"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.show()


def plot_search(oracles, df, lat_max=None, device=None, title=None):
    """Compare searches in the accuracy/latency plane.

    oracles: dict name -> Oracle (or list of Oracles, one per seed: the first is
             used for the scatter, all of them for the curves).
    Left:   accuracy vs latency of all architectures, of those evaluated by each
            method and of the best one found (star), with the Pareto front.
    Center: latency of the incumbent (best by objective so far) vs iterations.
    Right:  accuracy of the incumbent vs iterations.
    """
    acc, lat = df["acc"].to_numpy(), df["lat"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].scatter(lat, acc, s=2, alpha=0.15, color="lightgray", label="all")
    order = np.argsort(lat)
    pf = order[np.maximum.accumulate(acc[order]) == acc[order]]
    pf = pf[np.r_[True, np.diff(acc[pf]) > 0]]
    ax[0].plot(lat[pf], acc[pf], "k.-", lw=1, ms=3, label="Pareto front")
    for k, (name, os_) in enumerate(oracles.items()):
        os_ = os_ if isinstance(os_, (list, tuple)) else [os_]
        color = f"C{k}"
        q = os_[0].queried()
        ax[0].scatter(lat[q], acc[q], s=12, alpha=0.6, color=color, label=name)
        b = os_[0].incumbent_idx()[-1]
        ax[0].scatter(lat[b], acc[b], s=250, marker="*", color=color, edgecolor="k", zorder=5)
        inc = np.array([o.incumbent_idx() for o in os_])     # n_runs x n_queries
        x = np.arange(1, inc.shape[1] + 1)
        for a, vals in ((ax[1], lat[inc]), (ax[2], acc[inc])):
            m = vals.mean(0)
            a.plot(x, m, color=color, lw=2, label=name)
            if len(vals) > 1:
                se = vals.std(0) / np.sqrt(len(vals))
                a.fill_between(x, m - se, m + se, color=color, alpha=0.2)
    if lat_max is not None:
        for a in (ax[0], ax[1]):
            (a.axvline if a is ax[0] else a.axhline)(lat_max, color="r", ls="--", lw=1, label="LAT_MAX")
    unit = f"Latency on {device} (ms)" if device else "Latency (ms)"
    ax[0].set(xlabel=unit, ylabel="Accuracy (%)", title="Evaluated architectures (★ = best found)")
    ax[1].set(xlabel="# evaluated architectures", ylabel=unit, title="Latency of the best architecture so far")
    ax[2].set(xlabel="# evaluated architectures", ylabel="Accuracy (%)", title="Accuracy of the best architecture so far")
    ax[0].legend(fontsize=8, markerscale=1.5); ax[1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout(); plt.show()