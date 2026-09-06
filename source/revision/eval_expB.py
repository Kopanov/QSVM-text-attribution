#!/usr/bin/env python3
"""Experiment B: score the seven frozen configs on the new 1,000-text corpus.

No hyperparameter search. Transformers are fitted on the original training
split only, then applied to the new test texts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
EDITS = ROOT
RESULTS = ROOT / "results" / "generated"
OUT = RESULTS / "expB_testset"
CACHE = RESULTS / "_cache"
CORPUS = ROOT / "data"
TEST_DATA = CORPUS / "independent_test_corpus.csv"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SOURCE))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("FEATURE_MODE", "current")
os.environ.setdefault("QUANTUM_FEATURE_DIM", "14")

from run_revision_experiments import (  # noqa: E402
    GLOBAL_SEED,
    load_split,
    quantum_prepare,
    make_rbf,
    make_linearsvc_calibrated,
    per_class_metrics,
    _json_dump,
)

np.random.seed(GLOBAL_SEED)


def build_wide_csv() -> Path:
    """Return the released one-row-per-prompt independent-test corpus."""
    if not TEST_DATA.exists():
        raise SystemExit(f"Independent-test corpus not found: {TEST_DATA}")
    return TEST_DATA


def load_test_df(path: Path):
    import pandas as pd
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def transform_current(processor, df):
    texts, labels, cats = processor.extract_features_labels(df)
    y = processor.label_encoder.transform(labels)
    X_tfidf = processor.vectorizer.transform(texts)
    X_tfidf = X_tfidf.toarray() if hasattr(X_tfidf, "toarray") else np.asarray(X_tfidf)
    ling = processor.extract_linguistic_features(list(texts))
    ling = processor.feature_scaler.transform(ling)
    X = np.hstack([X_tfidf, ling])
    print(f"  current-mode test X={X.shape}  labels={list(processor.label_encoder.classes_)}")
    return X, np.asarray(y), texts, labels, cats


def transform_stylometric(processor, df):
    texts, labels, cats = processor.extract_features_labels(df)
    y = processor.label_encoder.transform(labels)
    ext = processor.extract_extended_linguistic_features(list(texts))
    X = processor.mi_selector.transform(ext)
    print(f"  stylometric_direct test X={X.shape}")
    return X, np.asarray(y), texts, labels, cats


def apply_quantum_map(qsvm, X):
    Xs = qsvm.feature_scaler.transform(X)
    if getattr(qsvm, "skip_pca", False) or qsvm.pca is None:
        Xp = Xs
    else:
        Xp = qsvm.pca.transform(Xs)
    return qsvm._normalize_to_2pi(Xp)


def save_config(rec: dict) -> None:
    rec["frozen"] = True
    rec["corpus"] = "data/independent_test_corpus.csv"
    rec["n_test"] = int(rec["n_test"])
    path = OUT / rec["filename"]
    _json_dump(path, {k: v for k, v in rec.items() if k != "filename"})
    acc = rec["test_accuracy"] * 100
    print(f"  saved {path.name}  acc={acc:.2f}%  n={rec['n_test']}")


def cfg7_global_linearsvc(Xtr, ytr, Xte, yte, classes):
    t0 = time.time()
    model = make_linearsvc_calibrated(1.0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    pc = per_class_metrics(yte, pred, classes)
    return {
        "config": 7,
        "filename": "config7_global_linearsvc.json",
        "name": "Global classical LinearSVC, full 3018-d",
        "train_n": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_accuracy": pc["accuracy"],
        "per_class": {k: v for k, v in pc.items() if k != "accuracy"},
        "seconds": time.time() - t0,
    }


def cfg6_pca_rbf(Xtr, ytr, Xte, yte, classes):
    t0 = time.time()
    qsvm, Xtr_q, Xte_q = quantum_prepare(Xtr, Xte, 14, skip_pca=False, reuse_saved=False)
    model = make_rbf(100.0)
    model.fit(Xtr_q, ytr)
    pred = model.predict(Xte_q)
    pc = per_class_metrics(yte, pred, classes)
    return {
        "config": 6,
        "filename": "config6_pca_csvm_d14.json",
        "name": "PCA+CSVM dim=14 RBF C=100 (best Experiment A)",
        "train_n": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_accuracy": pc["accuracy"],
        "per_class": {k: v for k, v in pc.items() if k != "accuracy"},
        "seconds": time.time() - t0,
    }


def cfg5_stylometric_csvm(Xtr, ytr, Xte, yte, classes):
    t0 = time.time()
    model = make_linearsvc_calibrated(1.0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    pc = per_class_metrics(yte, pred, classes)
    return {
        "config": 5,
        "filename": "config5_stylometric_csvm.json",
        "name": "Experiment 2.4 classical LinearSVC, 14 MI stylometric features",
        "train_n": int(len(ytr)),
        "n_test": int(len(yte)),
        "test_accuracy": pc["accuracy"],
        "per_class": {k: v for k, v in pc.items() if k != "accuracy"},
        "seconds": time.time() - t0,
        "n_features": int(Xtr.shape[1]),
    }


def qsvm_precomputed(Xtr_q, ytr, Xte_q, yte, classes, *, dim, C, train_n, name, filename,
                     k_train_path: Path | None, x_train_ref=None):
    from sklearn.svm import SVC
    from quantum_svm import QuantumSVM

    t0 = time.time()
    qsvm = QuantumSVM(
        feature_dim=dim,
        shots=1024,
        sample_size="full",
        quantum_backend="qiskit_statevector_deterministic",
        feature_map_reps=2,
        feature_map_type="ZZ",
        C=C,
        statevector_shots=None,
        skip_pca=True,  # X already quantum-mapped
    )
    qsvm.create_quantum_kernel()

    Ktr = None
    if k_train_path and k_train_path.exists():
        z = np.load(k_train_path)
        K_cached = z["K"].astype(np.float64)
        ok_shape = K_cached.shape == (len(Xtr_q), len(Xtr_q))
        xref = None
        if x_train_ref is not None:
            xref = np.asarray(x_train_ref)
        elif "X_train_q" in z.files:
            xref = z["X_train_q"]
        ok_x = False
        if xref is not None and xref.shape == Xtr_q.shape:
            ok_x = float(np.max(np.abs(xref - Xtr_q))) < 1e-6
        if ok_shape and ok_x:
            Ktr = K_cached
            print(f"    reused K_train {k_train_path.name} {Ktr.shape} (X_train_q matches)")
        else:
            print(f"    cache mismatch shape_ok={ok_shape} x_ok={ok_x}; recomputing K_train")
    if Ktr is None:
        print(f"    computing K_train {len(Xtr_q)}x{len(Xtr_q)} dim={dim} ...", flush=True)
        Ktr = np.asarray(qsvm.quantum_kernel.evaluate(x_vec=Xtr_q), dtype=np.float64)
        if k_train_path:
            np.savez_compressed(
                k_train_path,
                K=Ktr.astype(np.float32),
                y=ytr,
                X_train_q=np.asarray(Xtr_q, dtype=np.float32),
            )

    print(f"    computing K_test {len(Xte_q)}x{len(Xtr_q)} dim={dim} ...", flush=True)
    Kte = np.asarray(
        qsvm.quantum_kernel.evaluate(x_vec=Xte_q, y_vec=Xtr_q),
        dtype=np.float64,
    )
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(Ktr, ytr)
    pred = clf.predict(Kte)
    pc = per_class_metrics(yte, pred, classes)
    return {
        "config": None,
        "filename": filename,
        "name": name,
        "dim": dim,
        "C": C,
        "feature_map": "ZZFeatureMap",
        "reps": 2,
        "train_n": int(train_n),
        "n_test": int(len(yte)),
        "test_accuracy": pc["accuracy"],
        "per_class": {k: v for k, v in pc.items() if k != "accuracy"},
        "seconds": time.time() - t0,
        "kernel": "fidelity_statevector_exact",
    }


def run_classical(test_df) -> list[dict]:
    recs = []
    print("\n=== classical frozen evals (5, 6, 7) ===", flush=True)
    pack = load_split(3480, 1160, "current")
    Xtr, ytr = pack["X_train"], pack["y_train"]
    classes = pack["classes"]
    Xte, yte, *_ = transform_current(pack["processor"], test_df)

    rec = cfg7_global_linearsvc(Xtr, ytr, Xte, yte, classes)
    save_config(rec)
    recs.append(rec)

    rec = cfg6_pca_rbf(Xtr, ytr, Xte, yte, classes)
    save_config(rec)
    recs.append(rec)

    os.environ["FEATURE_MODE"] = "stylometric_direct"
    os.environ["QUANTUM_FEATURE_DIM"] = "14"
    pack_s = load_split(3480, 1160, "stylometric_direct")
    Xtr_s, ytr_s = pack_s["X_train"], pack_s["y_train"]
    Xte_s, yte_s, *_ = transform_stylometric(pack_s["processor"], test_df)
    rec = cfg5_stylometric_csvm(Xtr_s, ytr_s, Xte_s, yte_s, pack_s["classes"])
    save_config(rec)
    recs.append(rec)
    os.environ["FEATURE_MODE"] = "current"
    return recs


def run_qsvm(test_df) -> list[dict]:
    recs = []
    print("\n=== QSVM frozen evals (1, 2, 3, 4) ===", flush=True)

    # Configs 1–2: current-mode PCA QSVM, train 3480
    os.environ["FEATURE_MODE"] = "current"
    pack = load_split(3480, 1160, "current")
    Xtr, ytr = pack["X_train"], pack["y_train"]
    classes = pack["classes"]
    Xte, yte, *_ = transform_current(pack["processor"], test_df)

    for dim, cfg, fname, name in (
        (14, 1, "config1_qsvm_d14.json",
         "QSVM dim=14, C=1, ZZ reps=2, train 3480"),
        (12, 2, "config2_qsvm_d12.json",
         "QSVM dim=12, C=1, ZZ reps=2, train 3480"),
    ):
        print(f"\n-- config {cfg} dim={dim} --", flush=True)
        qsvm, Xtr_q, Xte_q = quantum_prepare(Xtr, Xte, dim, skip_pca=False, reuse_saved=False)
        cached = CACHE / f"quantum_inputs_d{dim}_3480.npz"
        if cached.exists():
            z = np.load(cached)
            if z["X_train_q"].shape == Xtr_q.shape:
                maxdiff = float(np.max(np.abs(z["X_train_q"] - Xtr_q)))
                print(f"    vs cache X_train_q max|diff|={maxdiff:.3e}")
        rec = qsvm_precomputed(
            Xtr_q, ytr, Xte_q, yte, classes,
            dim=dim, C=1.0, train_n=len(ytr), name=name, filename=fname,
            k_train_path=CACHE / f"K_expB_d{dim}_{len(ytr)}.npz",
            x_train_ref=None,
        )
        rec["config"] = cfg
        save_config(rec)
        recs.append(rec)

    # Config 3: sweet-spot dim=8, original 600/400 split
    print("\n-- config 3 dim=8 train 600 --", flush=True)
    pack8 = load_split(600, 400, "current")
    Xtr8, ytr8 = pack8["X_train"], pack8["y_train"]
    Xte8, yte8, *_ = transform_current(pack8["processor"], test_df)
    qsvm8, Xtr8_q, Xte8_q = quantum_prepare(Xtr8, Xte8, 8, skip_pca=False, reuse_saved=False)
    rec = qsvm_precomputed(
        Xtr8_q, ytr8, Xte8_q, yte8, pack8["classes"],
        dim=8, C=1.0, train_n=len(ytr8),
        name="QSVM dim=8, C=1, ZZ reps=2, train 600 (paper sweet-spot split)",
        filename="config3_qsvm_d8_600.json",
        k_train_path=CACHE / "K_expB_d8_600.npz",
        x_train_ref=None,
    )
    rec["config"] = 3
    save_config(rec)
    recs.append(rec)

    # Config 4: stylometric_direct QSVM, 14 MI features, train 3480
    print("\n-- config 4 stylometric_direct QSVM --", flush=True)
    os.environ["FEATURE_MODE"] = "stylometric_direct"
    os.environ["QUANTUM_FEATURE_DIM"] = "14"
    pack_s = load_split(3480, 1160, "stylometric_direct")
    Xtr_s, ytr_s = pack_s["X_train"], pack_s["y_train"]
    Xte_s, yte_s, *_ = transform_stylometric(pack_s["processor"], test_df)
    qsvm_s, Xtr_sq, Xte_sq = quantum_prepare(
        Xtr_s, Xte_s, 14, skip_pca=True, reuse_saved=False
    )
    rec = qsvm_precomputed(
        Xtr_sq, ytr_s, Xte_sq, yte_s, pack_s["classes"],
        dim=14, C=1.0, train_n=len(ytr_s),
        name="Experiment 2.3 QSVM, 14 MI stylometric features, C=1, ZZ reps=2",
        filename="config4_qsvm_stylometric.json",
        k_train_path=CACHE / "K_expB_stylometric_d14_3480.npz",
        x_train_ref=None,
    )
    rec["config"] = 4
    save_config(rec)
    recs.append(rec)
    os.environ["FEATURE_MODE"] = "current"
    return recs


def write_overview(recs: list[dict]) -> None:
    recs = sorted(recs, key=lambda r: r["config"])
    overview = {
        "n_test_texts": 1000,
        "n_prompts": 500,
        "frozen": True,
        "configs": [
            {
                "config": r["config"],
                "name": r["name"],
                "test_accuracy": r["test_accuracy"],
                "test_accuracy_pct": round(r["test_accuracy"] * 100, 2),
                "n_test": r["n_test"],
                "seconds": r.get("seconds"),
            }
            for r in recs
        ],
    }
    _json_dump(OUT / "overview.json", overview)
    corpus_meta = {}
    prev = OUT / "STATUS.json"
    if prev.exists():
        try:
            corpus_meta = json.loads(prev.read_text()).get("corpus") or {}
        except Exception:
            corpus_meta = {}
    if not corpus_meta:
        corpus_meta = {
            "n_texts": 1000,
            "n_per_model": 500,
            "n_prompts": 500,
            "same_prompts_both_models": True,
            "prompt_file": "corpus/prompts_expB_500.csv",
            "combined_jsonl": "test_corpus.jsonl",
            "gemma_csv": "corpus/gemma_seed42.csv",
            "qwen_csv": "corpus/qwen_seed42.csv",
            "disjoint_from_original_prompts": True,
            "seed": 42,
            "qwen_resampled_ids": ["B0154", "B0278", "B0281", "B0285", "B0445"],
        }
    _json_dump(OUT / "STATUS.json", {
        "status": "complete" if len(recs) >= 7 else "partial",
        "n_configs": len(recs),
        "overview": "overview.json",
        "corpus": corpus_meta,
        "note": (
            "All seven frozen configs scored on the 1,000-text new-prompt corpus. "
            "QSVM train kernels were recomputed from the same X_train_q used for K_test "
            "(Exp C K_quantum_d*_3480.npz were not reused: current PCA map differs)."
            if len(recs) >= 7
            else "Classical configs 5–7 are done; QSVM configs 1–4 still running."
        ),
    })
    print("\n=== Experiment B overview ===")
    for r in recs:
        print(f"  {r['config']}: {r['test_accuracy']*100:6.2f}%  {r['name']}")


def patch_insights(recs: list[dict]) -> None:
    if len(recs) < 7:
        return
    recs = sorted(recs, key=lambda r: r["config"])
    lines = [
        "## Experiment B — independent test set (R2.2)",
        "- Fresh corpus: **generated**. 500 new prompts (disjoint from `prompts_all.csv`), "
        "same list for both models → **1,000 texts**.",
        "- Frozen evaluation (no tuning) on that corpus:",
    ]
    for r in recs:
        lines.append(
            f"  - **{r['config']}** {r['name']}: **{r['test_accuracy']*100:.2f}%**"
        )
    lines.append(
        "- Files: `revision_results/expB_testset/` (`test_corpus.jsonl`, `overview.json`, config JSONs)."
    )
    block = "\n".join(lines) + "\n"

    for path in (RESULTS / "INSIGHTS.md", EDITS / "INSIGHTS.md"):
        if not path.exists():
            continue
        text = path.read_text()
        start = text.find("## Experiment B")
        if start < 0:
            path.write_text(text.rstrip() + "\n\n" + block)
            continue
        rest = text[start:]
        nxt = rest.find("\n## ", 1)
        end = start + nxt if nxt >= 0 else len(text)
        path.write_text(text[:start] + block + (text[end:] if nxt >= 0 else ""))
    print("Updated INSIGHTS.md Experiment B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "classical", "qsvm"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    fallback = OUT / "roc_fallback_not_used"
    fallback.mkdir(exist_ok=True)
    for name in ("config6_pca_csvm_d14.json", "config7_global_linearsvc.json"):
        src = OUT / name
        if src.exists() and "Not a newly generated corpus" in src.read_text():
            src.replace(fallback / name)
            print(f"moved old ROC fallback {name} -> roc_fallback_not_used/")

    wide = build_wide_csv()
    test_df = load_test_df(wide)
    print(f"test prompts={len(test_df)}")

    recs: list[dict] = []
    if args.only in ("all", "classical"):
        recs.extend(run_classical(test_df))
    if args.only in ("all", "qsvm"):
        recs.extend(run_qsvm(test_df))

    existing = []
    for p in sorted(OUT.glob("config*.json")):
        existing.append(json.loads(p.read_text()))
    if existing:
        by = {int(r["config"]): r for r in existing}
        recs = [by[k] for k in sorted(by)]
    write_overview(recs)
    patch_insights(recs)


if __name__ == "__main__":
    main()
