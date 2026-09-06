# Results

## Stage 1

`stage1/runs/` contains one compact JSON for each of the 83 benchmark configurations. `stage1/summary.csv` provides the principal configuration, accuracy, cross-validation, timing, and provenance fields in one file.

## Stage 2

`stage2/runs/` contains one compact JSON for each of the 12 feature-engineering sweep configurations. `stage2/summary.csv` is the corresponding index.

## Focused paper artifacts

- `feature_salience/` contains the PCA-only representation-salience records used for Table 7 and Figure 4. These scores do not use the QSVM decision function or kernel.
- `experiment_2_5.json` contains the matched classical RBF and linear controls reported in Table 11.
- `independent_test/` contains the seven frozen evaluations from Section 3.4 and Table 12.
- `kernel_concentration/overview.json` contains the RBF and quantum off-diagonal kernel statistics and PCA variance curve from Section 4.2.

## Sanitization

The Stage 1 and Stage 2 files were derived from the original `training_results_overview` JSONs. Hostnames, scheduler identifiers, cache filenames, resource time series, and local paths were removed. Experimental metrics, hyperparameters, split sizes, preprocessing settings, timing summaries, and software versions were retained.

Configuration tags containing `3480-2320` record the originally requested sample counts. The actual fields inside those JSONs show 3,480 training samples and the capped canonical holdout of 1,160 validation samples.

These results are licensed under [CC BY 4.0](../LICENSE-DATA). Please attribute the associated paper and indicate any modifications.
