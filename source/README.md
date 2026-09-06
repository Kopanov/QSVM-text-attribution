# Source code

The core pipeline is organized as follows:

- `qsvm_cli.py` — environment-variable-driven training entry point.
- `data_processor.py` — loading, prompt-level splitting, TF-IDF, linguistic features, scaling, and feature modes.
- `quantum_svm.py` — PCA, angle mapping, quantum feature maps, kernels, QSVC training, and cross-validation monitoring.
- `classical_svm.py` — calibrated LinearSVC baseline.
- `feature_importance.py` — PCA representation salience and classical-model importance.
- `quantum_svm_parallel_worker.py` — fold and batching helpers used by the quantum pipeline.
- `quantum_svm_cutensornet.py` — optional cuTensorNet backend.
- `tsne_visualizer.py` and `silhouette_analyzer.py` — representation diagnostics.
- `revision/run_revision_experiments.py` — matched classical controls and kernel-concentration analyses.
- `revision/eval_expB.py` — independent-test evaluation.

The public copy uses repository-relative data and output paths. Scheduler-only submission and monitoring utilities are intentionally excluded because they are infrastructure-specific and do not implement the scientific method.

The analytical behavior comes from the original experiment source. Release-only changes replace machine-specific paths and omit the serialization of host and scheduler identifiers.

The source code is licensed under the [Apache License 2.0](../LICENSE). Academic users are also requested to cite the associated paper using [`CITATION.cff`](../CITATION.cff).
