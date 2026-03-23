"""Profiling callback for OpenBoost training.

Instruments the training loop to produce structured JSON reports that
break down time by phase (histogram building, split finding, partitioning,
etc.). Designed for self-recursive improvement loops: profile → identify
bottleneck → optimize → re-profile → verify improvement.

Usage:
    # Explicit callback
    from openboost import GradientBoosting, ProfilingCallback
    profiler = ProfilingCallback(output_dir="logs/")
    model = GradientBoosting(n_trees=100)
    model.fit(X, y, callbacks=[profiler])
    print(profiler.report_path)

    # Environment variable (zero-code-change)
    OPENBOOST_PROFILE=1 uv run python train.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._callbacks import Callback, TrainingState


# =============================================================================
# Phase timer
# =============================================================================

class PhaseTimer:
    """Accumulates time for a named phase across multiple calls."""

    __slots__ = ("name", "_use_cuda", "_times", "_start")

    def __init__(self, name: str, use_cuda: bool = False):
        self.name = name
        self._use_cuda = use_cuda
        self._times: list[float] = []
        self._start: float | None = None

    def start(self) -> None:
        if self._use_cuda:
            from numba import cuda
            cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._use_cuda:
            from numba import cuda
            cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self._times.append(elapsed)
        self._start = None
        return elapsed

    @property
    def total(self) -> float:
        return sum(self._times)

    @property
    def count(self) -> int:
        return len(self._times)

    @property
    def mean(self) -> float:
        return self.total / self.count if self._times else 0.0

    def to_dict(self, total_time: float) -> dict:
        return {
            "total_s": round(self.total, 6),
            "pct": round(100 * self.total / total_time, 2) if total_time > 0 else 0,
            "calls": self.count,
            "mean_s": round(self.mean, 6),
        }


# =============================================================================
# Bottleneck recommendations
# =============================================================================

PHASE_RECOMMENDATIONS: dict[str, tuple[str, str]] = {
    "histogram_build": (
        "_backends/_cpu.py:build_histogram_cpu, _backends/_cuda.py:_build_histogram_shared_kernel",
        "shared-memory tiling, feature batching, reducing n_bins",
    ),
    "split_find": (
        "_core/_primitives.py:find_node_splits, _core/_split.py:find_best_split",
        "GPU parallel scan, vectorized prefix-sum split evaluation",
    ),
    "partition": (
        "_core/_primitives.py:partition_samples",
        "radix-sort-based partitioning, sorted index schemes",
    ),
    "gradient_compute": (
        "_loss.py loss functions",
        "fused GPU kernels, avoiding CPU-GPU copies for custom losses",
    ),
    "prediction_update": (
        "_models/_boosting.py prediction update loop",
        "fusing tree traversal + add, batching prediction updates",
    ),
    "leaf_values": (
        "_core/_primitives.py:compute_leaf_values",
        "GPU reduction kernel, batch leaf computation",
    ),
    "tree_overhead": (
        "_core/_tree.py:fit_tree, _core/_growth.py:LevelWiseGrowth.grow",
        "reduce Python overhead in growth loop, minimize object allocation",
    ),
    "grad_pred_loss": (
        "_models/_boosting.py training loop (loss_fn, tree predict, loss eval)",
        "fuse gradient+prediction, skip loss eval when no callbacks need it",
    ),
}


# =============================================================================
# Hardware info
# =============================================================================

def _collect_hardware_info() -> dict:
    info: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": None,
        "gpu": None,
        "gpu_memory_gb": None,
    }
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            info["ram_gb"] = round(int(result.stdout.strip()) / (1024**3), 1)
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["ram_gb"] = round(int(line.split()[1]) / (1024**2), 1)
                        break
        except Exception:
            pass
    try:
        from numba import cuda
        if cuda.is_available():
            dev = cuda.get_current_device()
            info["gpu"] = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
    except Exception:
        pass
    return info


def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


# =============================================================================
# Profiling callback
# =============================================================================

_PRIMITIVES_TO_WRAP = [
    "build_node_histograms",
    "find_node_splits",
    "partition_samples",
    "compute_leaf_values",
]

_PHASE_NAMES = {
    "build_node_histograms": "histogram_build",
    "find_node_splits": "split_find",
    "partition_samples": "partition",
    "compute_leaf_values": "leaf_values",
}

# Modules where fit_tree is imported and needs wrapping
_FIT_TREE_MODULES = [
    "openboost._core._tree",
    "openboost._models._boosting",
]


class ProfilingCallback(Callback):
    """Profile training phases and produce structured JSON reports.

    Wraps core primitive functions with timers during training to measure
    per-phase time breakdown. Writes a JSON report to output_dir on completion.

    Args:
        output_dir: Directory for profile JSON files. Created if missing.
        compare_last: If True, compare with the most recent previous profile.
    """

    def __init__(self, output_dir: str = "logs/", compare_last: bool = True):
        self.output_dir = Path(output_dir)
        self.compare_last = compare_last

        self._timers: dict[str, PhaseTimer] = {}
        self._tree_timers: list[dict[str, float]] = []
        self._round_start: float = 0.0
        self._round_phase_snapshot: dict[str, float] = {}
        self._train_start: float = 0.0
        self._originals: dict[str, Any] = {}
        self._use_cuda: bool = False

        self.report_path: Path | None = None
        self.report: dict | None = None

    def _get_timer(self, name: str) -> PhaseTimer:
        if name not in self._timers:
            self._timers[name] = PhaseTimer(name, use_cuda=self._use_cuda)
        return self._timers[name]

    # ----- wrapping / unwrapping -----

    def _wrap_primitives(self) -> None:
        import sys
        import openboost._core._primitives as prims_mod
        import openboost._core._growth as growth_mod

        # Wrap the 4 core primitives
        for func_name in _PRIMITIVES_TO_WRAP:
            original = getattr(prims_mod, func_name)
            self._originals[("prim", func_name)] = original
            phase_name = _PHASE_NAMES[func_name]
            timer = self._get_timer(phase_name)

            def make_wrapper(orig, tmr):
                def wrapper(*args, **kwargs):
                    tmr.start()
                    result = orig(*args, **kwargs)
                    tmr.stop()
                    return result
                return wrapper

            wrapped = make_wrapper(original, timer)
            setattr(prims_mod, func_name, wrapped)
            if hasattr(growth_mod, func_name):
                setattr(growth_mod, func_name, wrapped)

        # Wrap fit_tree to capture total tree-building time (includes orchestration)
        fit_tree_timer = self._get_timer("fit_tree")
        for mod_name in _FIT_TREE_MODULES:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "fit_tree"):
                original_ft = getattr(mod, "fit_tree")
                self._originals[("fit_tree", mod_name)] = original_ft
                wrapped_ft = make_wrapper(original_ft, fit_tree_timer)
                setattr(mod, "fit_tree", wrapped_ft)

    def _unwrap_primitives(self) -> None:
        import sys
        import openboost._core._primitives as prims_mod
        import openboost._core._growth as growth_mod

        for key, original in self._originals.items():
            kind, name = key
            if kind == "prim":
                setattr(prims_mod, name, original)
                if hasattr(growth_mod, name):
                    setattr(growth_mod, name, original)
            elif kind == "fit_tree":
                mod = sys.modules.get(name)
                if mod:
                    setattr(mod, "fit_tree", original)
        self._originals.clear()

    # ----- callback hooks -----

    def on_train_begin(self, state: TrainingState) -> None:
        from ._backends import is_cuda
        self._use_cuda = is_cuda()
        self._timers.clear()
        self._tree_timers.clear()
        self._wrap_primitives()
        self._train_start = time.perf_counter()

    def on_round_begin(self, state: TrainingState) -> None:
        self._round_start = time.perf_counter()
        self._round_phase_snapshot = {
            name: timer.total for name, timer in self._timers.items()
        }

    def on_round_end(self, state: TrainingState) -> bool:
        round_total = time.perf_counter() - self._round_start
        tree_entry: dict[str, float] = {"round": state.round_idx, "total_s": round_total}
        for name, timer in self._timers.items():
            prev = self._round_phase_snapshot.get(name, 0.0)
            tree_entry[f"{name}_s"] = round(timer.total - prev, 6)
        # Compute per-tree derived phases
        ft = tree_entry.get("fit_tree_s", 0)
        prims = sum(tree_entry.get(f"{p}_s", 0) for p in
                     ("histogram_build", "split_find", "partition", "leaf_values"))
        tree_entry["tree_overhead_s"] = round(max(0.0, ft - prims), 6)
        tree_entry["grad_pred_loss_s"] = round(max(0.0, round_total - ft), 6)
        self._tree_timers.append(tree_entry)
        return True

    def on_train_end(self, state: TrainingState) -> None:
        total_time = time.perf_counter() - self._train_start
        self._unwrap_primitives()

        # Compute derived phases from per-tree data
        # round_total = gradient_compute + fit_tree + prediction_update + loss_eval
        # fit_tree = primitives + orchestration_overhead
        total_round_time = sum(t["total_s"] for t in self._tree_timers)
        fit_tree_total = self._timers["fit_tree"].total if "fit_tree" in self._timers else 0
        # Time outside fit_tree but inside rounds = grad compute + pred update + loss eval
        outside_tree = max(0.0, total_round_time - fit_tree_total)
        # Primitives total
        prims_total = sum(
            self._timers[p].total for p in ("histogram_build", "split_find", "partition", "leaf_values")
            if p in self._timers
        )
        # Orchestration = fit_tree - primitives (Python overhead in growth strategies)
        orchestration = max(0.0, fit_tree_total - prims_total)

        # Build phases dict (show the most useful breakdown)
        phases = {}
        for name in ("histogram_build", "split_find", "partition", "leaf_values"):
            if name in self._timers:
                phases[name] = self._timers[name].to_dict(total_time)
        # Add fit_tree orchestration overhead
        if orchestration > 0:
            n_trees = self._timers["fit_tree"].count if "fit_tree" in self._timers else 0
            phases["tree_overhead"] = {
                "total_s": round(orchestration, 6),
                "pct": round(100 * orchestration / total_time, 2) if total_time > 0 else 0,
                "calls": n_trees,
                "mean_s": round(orchestration / n_trees, 6) if n_trees > 0 else 0,
            }
        # Add outside-tree time (gradient + prediction + loss eval)
        if outside_tree > 0:
            n_rounds = len(self._tree_timers)
            phases["grad_pred_loss"] = {
                "total_s": round(outside_tree, 6),
                "pct": round(100 * outside_tree / total_time, 2) if total_time > 0 else 0,
                "calls": n_rounds,
                "mean_s": round(outside_tree / n_rounds, 6) if n_rounds > 0 else 0,
            }
        # Other: time outside the training loop entirely (setup, teardown)
        accounted = fit_tree_total + outside_tree
        other_time = max(0.0, total_time - accounted)
        if other_time > 0.001:
            phases["other"] = {
                "total_s": round(other_time, 6),
                "pct": round(100 * other_time / total_time, 2) if total_time > 0 else 0,
                "calls": None,
                "mean_s": None,
            }

        # Bottlenecks: top 3 phases by pct (excluding "other")
        ranked = sorted(
            [(name, data) for name, data in phases.items() if name != "other"],
            key=lambda x: x[1]["pct"],
            reverse=True,
        )
        bottlenecks = []
        for rank, (phase, data) in enumerate(ranked[:3], 1):
            target, rec = PHASE_RECOMMENDATIONS.get(phase, ("unknown", "investigate"))
            bottlenecks.append({
                "rank": rank,
                "phase": phase,
                "pct": data["pct"],
                "target": target,
                "recommendation": rec,
            })

        # Dataset / model info
        model = state.model
        n_trees_actual = len(getattr(model, "trees_", []))
        dataset_info = {
            "n_samples": (model.X_binned_.n_samples
                          if getattr(model, "X_binned_", None) else None),
            "n_features": getattr(model, "n_features_in_", None),
            "n_trees": n_trees_actual,
            "max_depth": getattr(model, "max_depth", None),
            "learning_rate": getattr(model, "learning_rate", None),
            "loss": str(getattr(model, "loss", None)),
            "backend": "cuda" if self._use_cuda else "cpu",
        }

        report = {
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _get_git_sha(),
            "hardware": _collect_hardware_info(),
            "dataset": dataset_info,
            "total_time_s": round(total_time, 6),
            "phases": phases,
            "per_tree": self._tree_timers,
            "bottlenecks": bottlenecks,
        }

        # Comparison with previous run
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.compare_last:
            comparison = self._compare_with_previous(report)
            if comparison:
                report["comparison"] = comparison

        # Write report
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = self.output_dir / f"profile_{ts}.json"
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.report = report

    # ----- comparison -----

    def _compare_with_previous(self, current: dict) -> dict | None:
        existing = sorted(self.output_dir.glob("profile_*.json"))
        if not existing:
            return None
        prev_path = existing[-1]
        try:
            with open(prev_path) as f:
                prev = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        prev_total = prev.get("total_time_s", 0)
        cur_total = current["total_time_s"]

        comparison: dict[str, Any] = {
            "previous_run": str(prev_path),
            "delta_total_pct": _pct_delta(prev_total, cur_total),
            "phase_deltas": {},
        }
        for phase in current["phases"]:
            if phase in prev.get("phases", {}):
                prev_s = prev["phases"][phase]["total_s"]
                cur_s = current["phases"][phase]["total_s"]
                comparison["phase_deltas"][phase] = {
                    "previous_s": prev_s,
                    "current_s": cur_s,
                    "delta_pct": _pct_delta(prev_s, cur_s),
                }
        return comparison


def _pct_delta(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return round(100 * (new - old) / old, 2)


# =============================================================================
# Summary printer (machine-readable for improvement loops)
# =============================================================================

def print_profile_summary(report: dict) -> None:
    """Print a machine-readable summary of a profile report."""
    print("=== PROFILE SUMMARY ===")
    print(f"TOTAL: {report['total_time_s']:.2f}s")
    print(f"BACKEND: {report['dataset'].get('backend', 'unknown')}")

    if report.get("bottlenecks"):
        top = report["bottlenecks"][0]
        print(f"TOP BOTTLENECK: {top['phase']} ({top['pct']}%)")
        print(f"TARGET: {top['target']}")
        print(f"RECOMMENDATION: {top['recommendation']}")

    if report.get("comparison"):
        comp = report["comparison"]
        delta = comp["delta_total_pct"]
        sign = "+" if delta > 0 else ""
        print(f"DELTA vs PREVIOUS: {sign}{delta}% total")
        for phase, pd in comp.get("phase_deltas", {}).items():
            if abs(pd["delta_pct"]) >= 5:
                s = "+" if pd["delta_pct"] > 0 else ""
                print(f"  {phase}: {s}{pd['delta_pct']}%")
    else:
        print("DELTA vs PREVIOUS: (no previous run)")

    print(f"REPORT: {report.get('_path', 'N/A')}")
