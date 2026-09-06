#!/usr/bin/env python3
"""Revision experiments A/B/C. Reuses the original QuantumText_v2 pipeline.

Does not rewrite TF-IDF, linguistic features, splits, scaler, PCA, or [0, 2π]
normalization. Those come from data_processor.DataProcessor, quantum_svm.QuantumSVM,
classical_svm.ClassicalSVM, and the association helper from qsvm_cli.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
EDITS = ROOT
RESULTS = ROOT / "results" / "generated"
CACHE = RESULTS / "_cache"
DATA_FILE = ROOT / "data" / "main_corpus.csv"
GLOBAL_SEED = 42

sys.path.insert(0, str(SOURCE))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("FEATURE_MODE", "current")

np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


def _json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def calculate_model_association(X, y, feature_idx: int) -> dict:
    """Same rule as qsvm_cli.calculate_model_association (0=Gemma, 1=Qwen)."""
    feature_values = X[:, feature_idx]
    gemma_values = feature_values[y == 0]
    qwen_values = feature_values[y == 1]
    gemma_mean = float(np.mean(gemma_values)) if len(gemma_values) else 0.0
    qwen_mean = float(np.mean(qwen_values)) if len(qwen_values) else 0.0
    gemma_freq = float(np.mean(gemma_values > 0)) if len(gemma_values) else 0.0
    qwen_freq = float(np.mean(qwen_values > 0)) if len(qwen_values) else 0.0
    if gemma_mean > qwen_mean:
        preferred = "Gemma 3"
    else:
        preferred = "Qwen 2.5"
    confidence = abs(gemma_mean - qwen_mean) / max(gemma_mean + qwen_mean, 1e-8)
    return {
        "gemma_mean_value": gemma_mean,
        "qwen_mean_value": qwen_mean,
        "gemma_frequency": gemma_freq,
        "qwen_frequency": qwen_freq,
        "preferred_model": preferred,
        "confidence": float(min(confidence, 1.0)),
        "difference": float(abs(gemma_mean - qwen_mean)),
    }


def load_split(train_n: int, val_n: int, feature_mode: str = "current"):
    """Mirror qsvm_cli.py: optional early row subset, then DataProcessor split/slice."""
    from data_processor import DataProcessor

    os.environ["FEATURE_MODE"] = feature_mode
    processor = DataProcessor()
    processor.feature_mode = feature_mode
    df = processor.load_csv_with_flexible_delimiter(str(DATA_FILE))
    total_needed = train_n + val_n
    if total_needed > 0 and len(df) > total_needed:
        if "label" in df.columns:
            from sklearn.model_selection import train_test_split
            df, _ = train_test_split(
                df,
                train_size=total_needed,
                stratify=df["label"] if df["label"].nunique() > 1 else None,
                random_state=42,
            )
        else:
            df = df.head(total_needed)
        print(f"  subset df -> {len(df)} rows for {train_n}+{val_n}")

    data = processor.load_and_preprocess(df)
    X_train = data["X_train"][:train_n]
    y_train = data["y_train"][:train_n]
    X_val = data["X_val"][:val_n]
    y_val = data["y_val"][:val_n]
    classes = list(processor.label_encoder.classes_)
    print(f"  split {feature_mode}: X_train={X_train.shape} X_val={X_val.shape} classes={classes}")
    return {
        "processor": processor,
        "X_train": np.asarray(X_train),
        "y_train": np.asarray(y_train),
        "X_val": np.asarray(X_val),
        "y_val": np.asarray(y_val),
        "classes": classes,
        "feature_names": np.asarray(data["feature_names"]) if data.get("feature_names") is not None else np.array([]),
        "skip_pca": bool(data.get("skip_pca", False)),
        "mi_selected_feature_names": data.get("mi_selected_feature_names"),
        "texts_train": None,
        "texts_val": None,
    }


def quantum_prepare(X_train, X_val, dim: int, skip_pca: bool = False,
                    reuse_saved: bool = False):
    """StandardScaler → PCA(dim) → [0, 2π], fitted on train only (QuantumSVM).

    If reuse_saved and a published artifact exists for this (dim, 3480) split,
    load that PCA so the 14-d inputs are the ones the paper's QSVM consumed.
    The quantum StandardScaler is not stored separately; it is re-fit on the
    same 3,480-row matrix (already gate-verified against the paper).
    """
    from quantum_svm import QuantumSVM

    qsvm = QuantumSVM(
        feature_dim=dim,
        shots=1024,
        sample_size="full",
        quantum_backend="qiskit_statevector_deterministic",
        feature_map_reps=2,
        feature_map_type="ZZ",
        C=1.0,
        statevector_shots=None,
        skip_pca=skip_pca,
    )
    Xtr_q, Xva_q = qsvm._prepare_quantum_data(X_train, X_val)

    if reuse_saved and not skip_pca:
        import joblib
        art = {
            8: ROOT / "QSVM_stage1/QSVM_model_8-exact-600-400-Cs1_Cq1/inference_artifacts_8-exact-600-400-Cs1_Cq1",
            12: ROOT / "QSVM_stage1/QSVM_model_12-exact-3480-2320-Cs1_Cq1/inference_artifacts_12-exact-3480-2320-Cs1_Cq1",
            14: ROOT / "QSVM_stage1/QSVM_model_14-exact-3480-2320-Cs1_Cq1/inference_artifacts_14-exact-3480-2320-Cs1_Cq1",
            16: ROOT / "QSVM_stage1/QSVM_model_16-exact-3480-2320-Cs1_Cq1/inference_artifacts_16-exact-3480-2320-Cs1_Cq1",
        }.get(dim)
        pca_path = art / "QSVM_pca_quantum.joblib" if art else None
        if pca_path and pca_path.exists() and X_train.shape[0] >= 3000:
            saved = joblib.load(pca_path)
            if saved.n_components_ == dim:
                Xtr_s = qsvm.feature_scaler.transform(X_train)
                Xva_s = qsvm.feature_scaler.transform(X_val)
                Xtr_p = saved.transform(Xtr_s)
                Xva_p = saved.transform(Xva_s)
                qsvm.pca = saved
                qsvm.explained_variance = float(np.sum(saved.explained_variance_ratio_))
                norm_path = art / "QSVM_quantum_normalization_params.npz"
                if norm_path.exists():
                    nz = np.load(norm_path)
                    key_min = "x_min" if "x_min" in nz.files else ("quant_norm_min" if "quant_norm_min" in nz.files else nz.files[0])
                    key_max = "x_max" if "x_max" in nz.files else ("quant_norm_max" if "quant_norm_max" in nz.files else nz.files[1])
                    qsvm.quant_norm_min = np.asarray(nz[key_min])
                    qsvm.quant_norm_max = np.asarray(nz[key_max])
                    if qsvm.quant_norm_min.shape[0] != dim:
                        qsvm.quant_norm_min = Xtr_p.min(axis=0)
                        qsvm.quant_norm_max = Xtr_p.max(axis=0)
                else:
                    qsvm.quant_norm_min = Xtr_p.min(axis=0)
                    qsvm.quant_norm_max = Xtr_p.max(axis=0)
                Xtr_q = qsvm._normalize_to_2pi(Xtr_p)
                Xva_q = qsvm._normalize_to_2pi(Xva_p)
                print(f"    reused saved PCA+norm from {pca_path.parent.name}")
    return qsvm, np.asarray(Xtr_q), np.asarray(Xva_q)


def pca_attribution(pca, feature_names, X_orig_train, y_train, X_orig_val, y_val,
                    pca_space_weights=None) -> list:
    """Variance-weighted PCA reconstruction (feature_importance.py).

    If pca_space_weights is given (len=n_components), also weight by |w_i|.
    The returned `attribution` list uses the paper method (variance only).
    """
    evr = pca.explained_variance_ratio_
    loadings = np.abs(pca.components_)
    pca_importance = np.sum(loadings * evr[:, np.newaxis], axis=0)
    pca_sum = float(np.sum(pca_importance)) or 1.0
    scores = pca_importance / pca_sum * 100.0

    model_scores = None
    if pca_space_weights is not None:
        w = np.abs(np.asarray(pca_space_weights, dtype=float))
        w = w / (w.sum() or 1.0)
        weighted = np.sum(loadings * (evr * w)[:, np.newaxis], axis=0)
        wsum = float(np.sum(weighted)) or 1.0
        model_scores = weighted / wsum * 100.0

    names = list(feature_names)
    if len(names) != len(scores):
        names = [f"feature_{i}" for i in range(len(scores))]

    Xc = np.vstack([X_orig_train, X_orig_val])
    yc = np.hstack([y_train, y_val])
    order = np.argsort(-scores)
    rows = []
    for rank, idx in enumerate(order, start=1):
        assoc = calculate_model_association(Xc, yc, int(idx))
        rec = {
            "feature": names[idx],
            "score": float(scores[idx]),
            "model": assoc["preferred_model"],
            "rank": rank,
            "importance_score": float(scores[idx]),
            "model_association": assoc,
        }
        if model_scores is not None:
            rec["model_weighted_score"] = float(model_scores[idx])
        rows.append(rec)
    return rows


def per_class_metrics(y_true, y_pred, classes) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    out = {}
    for i, name in enumerate(classes):
        key = "Gemma 3" if str(name).lower().startswith("gemma") else "Qwen 2.5"
        out[key] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
        }
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    return out


def cv_refit_scaler_pca(X, y, dim: int, make_model, n_folds: int = 5) -> tuple[float, float]:
    """5-fold CV that refits StandardScaler+PCA+[0,2π] on each fold train (paper claim)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for tr, va in skf.split(X, y):
        qsvm, Xtr_q, Xva_q = quantum_prepare(X[tr], X[va], dim, skip_pca=False)
        model = make_model()
        model.fit(Xtr_q, y[tr])
        pred = model.predict(Xva_q)
        accs.append(float(accuracy_score(y[va], pred)))
        del qsvm
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)


def make_rbf(C: float):
    from sklearn.svm import SVC
    return SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", random_state=42)


def make_linearsvc_calibrated(C: float = 1.0):
    from classical_svm import ClassicalSVM
    csvm = ClassicalSVM(C=C)
    csvm.create_model()
    return csvm.model


def linear_pca_weights(model) -> np.ndarray | None:
    try:
        calibrated = model.calibrated_classifiers_[0]
        inner = getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", calibrated))
        if hasattr(inner, "coef_"):
            return np.abs(inner.coef_[0])
    except Exception:
        pass
    if hasattr(model, "coef_"):
        return np.abs(model.coef_[0])
    return None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_environment() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env_path = RESULTS / "environment.txt"
    if not env_path.exists():
        subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       check=False, stdout=open(env_path, "w"))
    versions = {"python": sys.version, "timestamp": _now()}
    for pkg in ("numpy", "scipy", "sklearn", "qiskit", "qiskit_machine_learning",
                "qiskit_aer", "pandas", "nltk", "joblib"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except Exception as e:
            versions[pkg] = f"error: {e}"
    _json_dump(RESULTS / "environment_versions.json", versions)
    print("Environment:", {k: versions[k] for k in ("python", "numpy", "sklearn", "qiskit")})
    return versions


def stage_reproduction_gate() -> dict:
    print("\n=== REPRODUCTION GATE ===")
    from classical_svm import ClassicalSVM
    from sklearn.metrics import accuracy_score

    # Gate 2 first (fast): LinearSVC full 3018-d, 3480/1160 → 98.02%
    print("Loading 3480/1160 current-mode split (full 3018-d classical)...")
    t0 = time.time()
    pack = load_split(3480, 1160, "current")
    print(f"  load {time.time()-t0:.1f}s")
    csvm = ClassicalSVM(C=1.0)
    csvm.train(pack["X_train"], pack["y_train"])
    pred, _ = csvm.predict(pack["X_val"])
    acc_csvm = float(accuracy_score(pack["y_val"], pred))
    print(f"  LinearSVC 3480/1160 val acc = {acc_csvm*100:.4f}%  (target 98.02%)")

    # Gate 1: QSVM dim=8, 600/400, C=1, ZZ reps=2 → 88.00%
    print("Loading 600/400 current-mode split and training QSVM dim=8...")
    pack8 = load_split(600, 400, "current")
    from quantum_svm import QuantumSVM
    qsvm = QuantumSVM(
        feature_dim=8, shots=1024, sample_size="full",
        quantum_backend="qiskit_statevector_deterministic",
        feature_map_reps=2, feature_map_type="ZZ",
        C=1.0, statevector_shots=None, skip_pca=False,
    )
    t1 = time.time()
    qres = qsvm.train_and_evaluate(pack8["X_train"], pack8["X_val"],
                                   pack8["y_train"], pack8["y_val"])
    acc_qsvm = float(qres.get("accuracy", qres.get("val_accuracy", -1)))
    print(f"  QSVM dim=8 600/400 val acc = {acc_qsvm*100:.4f}%  (target 88.00%)  [{time.time()-t1:.1f}s]")

    ok_c = abs(acc_csvm - 0.9801724137931035) < 1e-4 or abs(acc_csvm - 0.9802) < 5e-4
    # paper prints 98.02% = 0.980172...
    ok_c = abs(round(acc_csvm * 100, 2) - 98.02) < 0.02
    ok_q = abs(round(acc_qsvm * 100, 2) - 88.00) < 0.02
    gate = {
        "timestamp": _now(),
        "qsvm_dim8_600_400": {
            "accuracy": acc_qsvm,
            "accuracy_pct": round(acc_qsvm * 100, 4),
            "target_pct": 88.00,
            "pass": bool(ok_q),
            "seconds": time.time() - t1,
        },
        "csvm_full_3480_1160": {
            "accuracy": acc_csvm,
            "accuracy_pct": round(acc_csvm * 100, 4),
            "target_pct": 98.02,
            "pass": bool(ok_c),
        },
        "both_pass": bool(ok_c and ok_q),
        "numpy": np.__version__,
        "note": "Published runs used NumPy 1.26.4; this machine may differ.",
    }
    _json_dump(RESULTS / "reproduction_gate.json", gate)
    if not gate["both_pass"]:
        print("GATE FAILED — continuing to record the difference, but treat later numbers as not bitwise-comparable.")
    else:
        print("GATE PASSED")
    # persist the 3480 pack for later stages
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / "split_3480_1160_current.npz",
        X_train=pack["X_train"], y_train=pack["y_train"],
        X_val=pack["X_val"], y_val=pack["y_val"],
        feature_names=np.array(pack["processor"].vectorizer.get_feature_names_out().tolist()
                               + pack["processor"].get_linguistic_feature_names(), dtype=object),
        classes=np.array(pack["classes"], dtype=object),
    )
    np.savez_compressed(
        CACHE / "split_600_400_current.npz",
        X_train=pack8["X_train"], y_train=pack8["y_train"],
        X_val=pack8["X_val"], y_val=pack8["y_val"],
        classes=np.array(pack8["classes"], dtype=object),
    )
    return gate


def _load_cached_3480():
    p = CACHE / "split_3480_1160_current.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        return {
            "X_train": z["X_train"], "y_train": z["y_train"],
            "X_val": z["X_val"], "y_val": z["y_val"],
            "feature_names": z["feature_names"],
            "classes": list(z["classes"]),
        }
    pack = load_split(3480, 1160, "current")
    names = np.array(
        pack["processor"].vectorizer.get_feature_names_out().tolist()
        + pack["processor"].get_linguistic_feature_names(),
        dtype=object,
    )
    return {
        "X_train": pack["X_train"], "y_train": pack["y_train"],
        "X_val": pack["X_val"], "y_val": pack["y_val"],
        "feature_names": names,
        "classes": pack["classes"],
    }


def stage_expA() -> dict:
    print("\n=== EXPERIMENT A — PCA + Classical SVM ===")
    pack = _load_cached_3480()
    Xtr, ytr, Xva, yva = pack["X_train"], pack["y_train"], pack["X_val"], pack["y_val"]
    names = pack["feature_names"]
    classes = pack["classes"]
    out_dir = RESULTS / "expA_pca_csvm"
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = [8, 10, 12, 14, 16]
    specs = [("rbf", 1.0), ("rbf", 10.0), ("rbf", 100.0), ("linearsvc", 1.0)]
    summary = []

    for dim in dims:
        print(f"\n-- PCA dim={dim} --")
        qsvm, Xtr_q, Xva_q = quantum_prepare(Xtr, Xva, dim, skip_pca=False, reuse_saved=False)
        pca = qsvm.pca
        evr = [float(x) for x in pca.explained_variance_ratio_]
        np.savez_compressed(
            CACHE / f"quantum_inputs_d{dim}_3480.npz",
            X_train_q=Xtr_q, X_val_q=Xva_q, y_train=ytr, y_val=yva,
            evr=np.array(evr),
            quant_norm_min=qsvm.quant_norm_min,
            quant_norm_max=qsvm.quant_norm_max,
        )

        for kernel, C in specs:
            tag = f"d{dim}_{kernel}_C{C:g}"
            print(f"  training {tag} ...", flush=True)
            t0 = time.time()
            if kernel == "rbf":
                model = make_rbf(C)
                model.fit(Xtr_q, ytr)
                pred = model.predict(Xva_q)
                w = None
                if hasattr(model, "dual_coef_") and hasattr(model, "support_vectors_"):
                    # effective linear weights in PCA/[0,2π] space (not the RBF feature map)
                    w = np.abs(model.dual_coef_[0] @ model.support_vectors_)
            else:
                model = make_linearsvc_calibrated(1.0)
                model.fit(Xtr_q, ytr)
                pred = model.predict(Xva_q)
                w = linear_pca_weights(model)

            pc = per_class_metrics(yva, pred, classes)
            cv_mean, cv_std = cv_refit_scaler_pca(
                Xtr, ytr, dim,
                (lambda C=C, kernel=kernel: make_rbf(C) if kernel == "rbf"
                 else make_linearsvc_calibrated(1.0)),
            )
            attrib = pca_attribution(pca, names, Xtr, ytr, Xva, yva, pca_space_weights=w)
            rec = {
                "dim": dim,
                "kernel": kernel,
                "C": C,
                "val_accuracy": pc["accuracy"],
                "per_class": {k: v for k, v in pc.items() if k != "accuracy"},
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "pca_explained_variance_sum": float(sum(evr)),
                "pca_explained_variance_ratio": evr,
                "seconds": time.time() - t0,
                "attribution": [
                    {"feature": a["feature"], "score": a["score"],
                     "model": a["model"], "rank": a["rank"]}
                    for a in attrib
                ],
                "attribution_full": attrib[:200],
            }
            _json_dump(out_dir / f"{tag}.json", rec)
            summary.append({
                "tag": tag, "dim": dim, "kernel": kernel, "C": C,
                "val_accuracy": rec["val_accuracy"],
                "cv_mean": cv_mean, "cv_std": cv_std,
            })
            print(f"    val={rec['val_accuracy']*100:.2f}%  cv={cv_mean*100:.2f}±{cv_std*100:.2f}  [{rec['seconds']:.1f}s]")

    # pick best at dim=14 (primary comparison) and globally
    dim14 = [s for s in summary if s["dim"] == 14]
    best14 = max(dim14, key=lambda s: s["val_accuracy"])
    best_all = max(summary, key=lambda s: s["val_accuracy"])
    overview = {
        "runs": summary,
        "best_dim14": best14,
        "best_overall": best_all,
        "qsvm_dim14_reference": 0.8801724137931034,
        "qsvm_dim8_600_reference": 0.88,
    }
    _json_dump(out_dir / "overview.json", overview)
    return overview


def stage_expC() -> dict:
    print("\n=== EXPERIMENT C — ceiling diagnostics ===")
    pack = _load_cached_3480()
    Xtr, ytr, Xva, yva = pack["X_train"], pack["y_train"], pack["X_val"], pack["y_val"]
    out = RESULTS / "expC_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    # C.1 PCA information loss on the 3018-d training vectors (after quantum StandardScaler)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    pca_full = PCA(random_state=42)
    pca_full.fit(Xtr_s)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    curve = {
        "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]),
        "cumulative_at": {str(d): float(cum[d - 1]) for d in (8, 10, 12, 14, 16, 32, 64, 128, 256)},
        "cumulative_first_64": [float(x) for x in cum[:64]],
        "explained_variance_ratio_first_64": [float(x) for x in evr[:64]],
        "n_components_for_50pct": int(np.searchsorted(cum, 0.50) + 1),
        "n_components_for_80pct": int(np.searchsorted(cum, 0.80) + 1),
        "n_components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
    }
    _json_dump(out / "pca_explained_variance.json", curve)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(1, min(256, len(cum)) + 1)
    ax.plot(xs, cum[: len(xs)] * 100, color="#1f4e79", lw=2)
    for d in (8, 10, 12, 14, 16):
        ax.axvline(d, color="#999", ls=":", lw=0.8)
        ax.scatter([d], [cum[d - 1] * 100], zorder=3)
        ax.annotate(f"d={d}\n{cum[d-1]*100:.1f}%", (d, cum[d - 1] * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pca_explained_variance.png", dpi=200)
    plt.close(fig)
    print("  C.1 PCA variance:", {d: f"{curve['cumulative_at'][str(d)]*100:.2f}%"
                                  for d in (8, 10, 12, 14, 16)})

    # C.2 classical RBF kernel stats on identical 14-d quantum inputs (cheap)
    z14 = CACHE / "quantum_inputs_d14_3480.npz"
    if not z14.exists():
        qsvm, Xtr_q, Xva_q = quantum_prepare(Xtr, Xva, 14)
        np.savez_compressed(z14, X_train_q=Xtr_q, X_val_q=Xva_q, y_train=ytr, y_val=yva)
    else:
        z = np.load(z14)
        Xtr_q, ytr = z["X_train_q"], z["y_train"]
        Xva_q = z["X_val_q"]

    from sklearn.metrics.pairwise import rbf_kernel
    K_rbf = rbf_kernel(Xtr_q, Xtr_q, gamma=None)  # gamma='scale' ≡ 1/(n_features * X.var())
    off = K_rbf[~np.eye(K_rbf.shape[0], dtype=bool)]
    rbf_stats = {
        "kernel": "rbf",
        "gamma": "scale",
        "dim": 14,
        "n": int(K_rbf.shape[0]),
        "offdiag_mean": float(off.mean()),
        "offdiag_var": float(off.var()),
        "offdiag_std": float(off.std()),
        "diag_mean": float(np.diag(K_rbf).mean()),
    }
    _json_dump(out / "kernel_rbf_d14_stats.json", rbf_stats)
    np.savez_compressed(CACHE / "K_rbf_d14_3480.npz", K=K_rbf.astype(np.float32), y=ytr)
    print(f"  C.2 RBF off-diag mean={rbf_stats['offdiag_mean']:.6f} var={rbf_stats['offdiag_var']:.6e}")

    # C.2/C.3 quantum kernels: compute if missing (expensive). dim=8 first, then 12, 14, 16.
    q_stats = {}
    for dim in (8, 12, 14, 16):
        stats_path = out / f"kernel_quantum_d{dim}_stats.json"
        k_path = CACHE / f"K_quantum_d{dim}_3480.npz"
        if stats_path.exists() and k_path.exists():
            q_stats[dim] = json.loads(stats_path.read_text())
            print(f"  C.2 quantum d={dim} reused")
            continue
        print(f"  C.2 computing quantum kernel dim={dim} on {len(Xtr)} train ...", flush=True)
        t0 = time.time()
        qsvm, Xtr_qd, _ = quantum_prepare(Xtr, Xva, dim, reuse_saved=False)
        qsvm.create_quantum_kernel()
        Kq = np.asarray(qsvm.quantum_kernel.evaluate(x_vec=Xtr_qd), dtype=np.float64)
        offq = Kq[~np.eye(Kq.shape[0], dtype=bool)]
        rec = {
            "kernel": "fidelity_statevector_exact",
            "feature_map": "ZZFeatureMap",
            "reps": 2,
            "dim": dim,
            "n": int(Kq.shape[0]),
            "offdiag_mean": float(offq.mean()),
            "offdiag_var": float(offq.var()),
            "offdiag_std": float(offq.std()),
            "diag_mean": float(np.diag(Kq).mean()),
            "seconds": time.time() - t0,
        }
        _json_dump(stats_path, rec)
        np.savez_compressed(k_path, K=Kq.astype(np.float32), y=ytr)
        q_stats[dim] = rec
        print(f"    d={dim} off-diag mean={rec['offdiag_mean']:.6f} var={rec['offdiag_var']:.6e} [{rec['seconds']:.1f}s]")

    # C.3 kernel-PCA embeddings for dim=14 quantum vs RBF
    from sklearn.decomposition import KernelPCA
    k14 = CACHE / "K_quantum_d14_3480.npz"
    embed = {}
    if k14.exists():
        Kq = np.load(k14)["K"].astype(np.float64)
        for name, K in (("quantum_zz", Kq), ("rbf", K_rbf.astype(np.float64))):
            kpca = KernelPCA(n_components=2, kernel="precomputed", random_state=42)
            xy = kpca.fit_transform(K)
            embed[name] = {
                "x": [float(v) for v in xy[:, 0]],
                "y": [float(v) for v in xy[:, 1]],
                "label": ["Gemma" if int(v) == 0 else "Qwen" for v in ytr],
            }
            import csv
            csv_path = out / f"kpca_{name}_d14.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "label"])
                for i in range(len(xy)):
                    w.writerow([xy[i, 0], xy[i, 1], embed[name]["label"][i]])
        _json_dump(out / "kpca_d14_coordinates.json", embed)

        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=False, sharey=False)
        for ax, name, title in (
            (axes[0], "quantum_zz", "Quantum fidelity kernel (dim=14)"),
            (axes[1], "rbf", "Classical RBF kernel (same 14-d inputs)"),
        ):
            xs = np.array(embed[name]["x"])
            ys = np.array(embed[name]["y"])
            labs = np.array(embed[name]["label"])
            for lab, col in (("Gemma", "#d95f02"), ("Qwen", "#1b9e77")):
                m = labs == lab
                ax.scatter(xs[m], ys[m], s=8, alpha=0.45, c=col, label=lab, linewidths=0)
            ax.legend(fontsize=8, markerscale=1.6)
            ax.set_xlabel("kPCA-1")
            ax.set_ylabel("kPCA-2")
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "kpca_quantum_vs_rbf_d14.png", dpi=200)
        plt.close(fig)
        print("  C.3 kernel-PCA figure saved")
    else:
        print("  C.3 skipped (dim=14 quantum kernel not yet available)")

    overview = {"pca": curve, "rbf_d14": rbf_stats, "quantum": q_stats}
    _json_dump(out / "overview.json", overview)
    return overview


def stage_expB() -> dict:
    print("\n=== EXPERIMENT B — untouched test set ===")
    out = RESULTS / "expB_testset"
    out.mkdir(parents=True, exist_ok=True)

    note = {
        "generation_attempted": True,
        "generation_possible": False,
        "reason": (
            "No original generation script is in this repository, no Gemma 3 / Qwen 2.5 "
            "checkpoints are available on this machine, and there is no GPU driver. "
            "A fresh 1,000-sample new-prompt corpus cannot be generated here."
        ),
        "honest_fallback": None,
    }

    # Inspect the released independent-test CSV for prompt overlap with the training corpus.
    from data_processor import DataProcessor
    proc = DataProcessor()
    train_df = proc.load_csv_with_flexible_delimiter(str(DATA_FILE))
    train_prompts = set(train_df["prompt"].astype(str).str.strip())
    test_path = ROOT / "data" / "independent_test_corpus.csv"
    fallback = None
    if test_path.exists():
        import pandas as pd
        test_df = pd.read_csv(test_path)
        test_prompts = set(test_df["prompt"].astype(str).str.strip()) if "prompt" in test_df.columns else set()
        overlap = train_prompts & test_prompts
        fallback = {
            "file": str(test_path),
            "n_rows": int(len(test_df)),
            "n_test_prompts": int(len(test_prompts)),
            "n_train_prompts": int(len(train_prompts)),
            "n_overlapping_prompts": int(len(overlap)),
            "usable_as_disjoint_test": bool(len(overlap) == 0 and len(test_prompts) > 0),
        }
        note["roc_input_inspection"] = fallback
        print(f"  ROC input: {fallback}")

        if fallback["usable_as_disjoint_test"]:
            note["honest_fallback"] = (
                "Existing 00_QuantumROC input CSV has zero prompt overlap with the "
                "training corpus. Evaluating frozen classical configs on it. QSVM "
                "evals on this file are optional/expensive and recorded if completed."
            )
            _eval_frozen_classical_on_external(test_df, out)
        else:
            note["honest_fallback"] = (
                "The only extra CSV is not a clean disjoint test set "
                f"(overlap={fallback['n_overlapping_prompts']} prompts). "
                "No Experiment B numbers are reported as unbiased test accuracy."
            )

    _json_dump(out / "generation_config.json", note)
    _json_dump(out / "STATUS.json", {
        "status": "incomplete_no_fresh_corpus",
        "detail": note,
    })
    return note


def _eval_frozen_classical_on_external(test_df, out: Path) -> None:
    """Apply train-fitted current-mode transformers to an external paired CSV."""
    from sklearn.metrics import accuracy_score
    pack = _load_cached_3480()
    # Rebuild processor on the same 3480/1160 split so transformers match
    train_pack = load_split(3480, 1160, "current")
    processor = train_pack["processor"]
    texts, labels, _ = processor.extract_features_labels(test_df)
    if not texts:
        return
    y = processor.label_encoder.transform(labels)
    X_tfidf = processor.vectorizer.transform(texts)
    X_tfidf = X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else np.asarray(X_tfidf)
    ling = processor.extract_linguistic_features(list(texts))
    ling = processor.feature_scaler.transform(ling)
    X = np.hstack([X_tfidf, ling])

    from classical_svm import ClassicalSVM
    csvm = ClassicalSVM(C=1.0)
    csvm.train(train_pack["X_train"], train_pack["y_train"])
    pred, _ = csvm.predict(X)
    rec7 = {
        "config": 7,
        "name": "Global classical LinearSVC, full 3018-d",
        "n_test": int(len(y)),
        "test_accuracy": float(accuracy_score(y, pred)),
        "per_class": per_class_metrics(y, pred, train_pack["classes"]),
        "warning": "Not a newly generated corpus. See generation_config.json.",
    }
    _json_dump(out / "config7_global_linearsvc.json", rec7)

    # config 6: PCA+CSVM dim=14 best kernel from Exp A if available
    overview_a = RESULTS / "expA_pca_csvm" / "overview.json"
    kernel, C = "rbf", 1.0
    if overview_a.exists():
        best = json.loads(overview_a.read_text())["best_dim14"]
        kernel, C = best["kernel"], best["C"]
    qsvm, Xtr_q, _ = quantum_prepare(train_pack["X_train"], train_pack["X_val"], 14)
    Xte_scaled = qsvm.feature_scaler.transform(X)
    Xte_pca = qsvm.pca.transform(Xte_scaled)
    Xte_q = qsvm._normalize_to_2pi(Xte_pca)
    if kernel == "rbf":
        model = make_rbf(C)
        model.fit(Xtr_q, train_pack["y_train"])
    else:
        model = make_linearsvc_calibrated(1.0)
        model.fit(Xtr_q, train_pack["y_train"])
    pred6 = model.predict(Xte_q)
    rec6 = {
        "config": 6,
        "name": f"PCA+CSVM dim=14 {kernel} C={C}",
        "n_test": int(len(y)),
        "test_accuracy": float(accuracy_score(y, pred6)),
        "per_class": per_class_metrics(y, pred6, train_pack["classes"]),
        "warning": "Not a newly generated corpus. See generation_config.json.",
    }
    _json_dump(out / "config6_pca_csvm_d14.json", rec6)
    print(f"  fallback cfg7 acc={rec7['test_accuracy']*100:.2f}%  cfg6 acc={rec6['test_accuracy']*100:.2f}%")


def stage_grab() -> None:
    print("\n=== GRAB existing artifacts ===")
    dest = RESULTS / "grabbed"
    dest.mkdir(parents=True, exist_ok=True)

    # Stage 2 12-run sweep overviews
    s2 = dest / "stage2_sweep"
    s2.mkdir(exist_ok=True)
    stage2 = ROOT / "QSVM_stage2"
    if stage2.exists():
        for p in sorted(stage2.glob("QSVM_model_*/training_results_overview_*.json")):
            shutil.copy2(p, s2 / f"{p.parent.name}__{p.name}")

    # QSVM feature-importance (named linguistic) for dims 8,10,12,14,16
    ladder = {
        8: ROOT / "QSVM_stage1/QSVM_model_8-exact-600-400-Cs1_Cq1",
        10: ROOT / "QSVM_stage1/QSVM_model_10-exact-1600-600-Cs1_Cq1",
        12: ROOT / "QSVM_stage1/QSVM_model_12-exact-3480-2320-Cs1_Cq1",
        14: ROOT / "QSVM_stage1/QSVM_model_14-exact-3480-2320-Cs1_Cq1",
        16: ROOT / "QSVM_stage1/QSVM_model_16-exact-3480-2320-Cs1_Cq1",
    }
    fi_dir = dest / "qsvm_feature_importance_ladder"
    fi_dir.mkdir(exist_ok=True)
    for dim, folder in ladder.items():
        full = next(folder.glob("training_results_*.json"), None)
        if full is None:
            continue
        # skip overview
        cands = [p for p in folder.glob("training_results_*.json") if "overview" not in p.name]
        if not cands:
            continue
        data = json.loads(cands[0].read_text())
        qfi = data.get("quantum", {}).get("feature_importance_analysis", {})
        slim = {
            "dim": dim,
            "source": str(cands[0]),
            "analysis_method": qfi.get("analysis_method") or qfi.get("method"),
            "top_100_features": qfi.get("top_100_features", []),
            "all_6_lexical_features": qfi.get("all_6_lexical_features"),
            "all_6_syntactic_features": qfi.get("all_6_syntactic_features"),
            "all_6_stylistic_features": qfi.get("all_6_stylistic_features"),
        }
        _json_dump(fi_dir / f"qsvm_dim{dim}_importance.json", slim)

    # plotting script pointer
    (dest / "figure_scripts.txt").write_text(
        "docx_build/render_figures.py  — regenerates Figures 1–6 from Stage 1 JSONs\n"
        "Experiment-guide request: larger fonts, no embedded titles.\n"
    )
    print("  grabbed Stage 2 overviews and QSVM attribution ladder")


def write_insights(gate, expA, expC, expB) -> None:
    lines = []
    lines.append("# Revision experiment insights (Round 1)")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("All numbers below reuse the original `DataProcessor` / `QuantumSVM` / `ClassicalSVM` code,")
    lines.append("corpus `Qwen2.5-and-Gemma3-train.csv`, seed 42, and the same train/val slicing as `qsvm_cli.py`.")
    lines.append("")

    lines.append("## Environment")
    lines.append(f"- Python / Qiskit / sklearn match the paper. NumPy on this machine: **{np.__version__}**")
    lines.append("  (published Table A1: 1.26.4). Full `pip freeze` is in `revision_results/environment.txt`.")
    lines.append("")

    lines.append("## Reproduction gate")
    if gate:
        q = gate.get("qsvm_dim8_600_400", {})
        c = gate.get("csvm_full_3480_1160", {})
        lines.append(f"- QSVM dim=8, 600/400, C=1: **{q.get('accuracy_pct')}%** (target 88.00%) — {'PASS' if q.get('pass') else 'FAIL'}")
        lines.append(f"- LinearSVC full 3,018-d, 3,480/1,160: **{c.get('accuracy_pct')}%** (target 98.02%) — {'PASS' if c.get('pass') else 'FAIL'}")
        if not gate.get("both_pass"):
            lines.append("- Gate did not fully pass. Later comparisons are still informative but not bitwise-identical to the paper.")
    lines.append("")

    lines.append("## Experiment A — PCA + classical SVM (R2.5, R2.7, R2.10)")
    if expA:
        lines.append("Classical models trained on the *identical* QSVM inputs (StandardScaler → PCA → [0, 2π]).")
        lines.append("")
        lines.append("| dim | kernel | C | val acc | 5-fold CV |")
        lines.append("|-----|--------|---|---------|-----------|")
        for s in expA.get("runs", []):
            lines.append(
                f"| {s['dim']} | {s['kernel']} | {s['C']} | "
                f"{s['val_accuracy']*100:.2f}% | {s['cv_mean']*100:.2f}±{s['cv_std']*100:.2f} |"
            )
        b = expA.get("best_dim14")
        if b:
            gap = (b["val_accuracy"] - 0.8801724137931034) * 100
            lines.append("")
            lines.append(
                f"Best at dim=14: **{b['kernel']} C={b['C']} → {b['val_accuracy']*100:.2f}%** "
                f"(QSVM on the same 14-d input: 88.02%; gap = {gap:+.2f} pp)."
            )
        lines.append("")
        lines.append("Attribution note: the paper's QSVM importance is *only* variance-weighted PCA reconstruction.")
        lines.append("Applied unchanged, that profile is a property of the PCA, not of QSVM vs CSVM.")
        lines.append("`attribution` in each JSON is that PCA profile; `model_weighted_score` additionally")
        lines.append("weights components by |classifier weights| in the 14-d space so the two models can differ.")
    lines.append("")

    lines.append("## Experiment C — why ~88% (R2.9, R2.10.3)")
    if expC and "pca" in expC:
        at = expC["pca"]["cumulative_at"]
        lines.append("Cumulative explained variance of the 3,018-d training matrix (after the quantum StandardScaler):")
        for d in (8, 10, 12, 14, 16):
            lines.append(f"- d={d}: **{float(at[str(d)])*100:.2f}%** of variance retained")
        lines.append(
            f"- Components needed for 80/90/95% variance: "
            f"{expC['pca']['n_components_for_80pct']} / "
            f"{expC['pca']['n_components_for_90pct']} / "
            f"{expC['pca']['n_components_for_95pct']}"
        )
        lines.append("PCA to 14 components is a severe information bottleneck — this is a quantitative")
        lines.append("ceiling cause, not a speculation.")
        if expC.get("rbf_d14"):
            r = expC["rbf_d14"]
            lines.append(
                f"- RBF kernel (same 14-d inputs) off-diagonal mean={r['offdiag_mean']:.4f}, var={r['offdiag_var']:.4e}"
            )
        if expC.get("quantum"):
            lines.append("- Quantum fidelity-kernel off-diagonal stats:")
            for d, rec in sorted(expC["quantum"].items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"  - d={d}: mean={rec['offdiag_mean']:.4f}, var={rec['offdiag_var']:.4e}"
                )
    lines.append("")

    lines.append("## Experiment B — independent test set (R2.2)")
    if expB:
        lines.append(f"- Fresh corpus: **not generated** ({expB.get('reason','')})")
        if expB.get("honest_fallback"):
            lines.append(f"- {expB['honest_fallback']}")
    lines.append("")

    lines.append("## What this unlocks in the paper")
    lines.append("1. Replace the 'identical conditions' wording with: Stage 1 CSVM is a *global* 3,018-d upper bound;")
    lines.append("   the like-for-like kernel comparison is PCA+CSVM vs PCA+QSVM (Experiment A).")
    lines.append("2. Soften or drop the sentence Reviewer 2 quoted in R2.10 until kernel-PCA (C.3) and the")
    lines.append("   model-weighted fingerprints are discussed honestly.")
    lines.append("3. Ground the 88% ceiling in the explained-variance numbers (C.1) plus kernel concentration (C.2).")
    lines.append("4. R2.2 remains open until a new-prompt test corpus can be generated.")
    lines.append("")

    text = "\n".join(lines)
    (RESULTS / "INSIGHTS.md").write_text(text)
    (EDITS / "INSIGHTS.md").write_text(text)
    print("Wrote INSIGHTS.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "env", "gate", "A", "C", "B", "grab", "insights"])
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    versions = stage_environment() if args.stage in ("all", "env", "insights") else None
    gate = None
    expA = None
    expC = None
    expB = None

    if args.stage in ("all", "gate"):
        gate = stage_reproduction_gate()
    elif (RESULTS / "reproduction_gate.json").exists():
        gate = json.loads((RESULTS / "reproduction_gate.json").read_text())

    if args.stage in ("all", "A"):
        expA = stage_expA()
    elif (RESULTS / "expA_pca_csvm/overview.json").exists():
        expA = json.loads((RESULTS / "expA_pca_csvm/overview.json").read_text())

    if args.stage in ("all", "C"):
        expC = stage_expC()
    elif (RESULTS / "expC_diagnostics/overview.json").exists():
        expC = json.loads((RESULTS / "expC_diagnostics/overview.json").read_text())

    if args.stage in ("all", "B"):
        expB = stage_expB()
    elif (RESULTS / "expB_testset/generation_config.json").exists():
        expB = json.loads((RESULTS / "expB_testset/generation_config.json").read_text())

    if args.stage in ("all", "grab"):
        stage_grab()

    if args.stage in ("all", "insights", "A", "C", "B", "gate"):
        write_insights(gate, expA, expC, expB)

    print("\nDone. Results in", RESULTS)


if __name__ == "__main__":
    main()
