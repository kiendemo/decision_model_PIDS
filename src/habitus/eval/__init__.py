"""Evaluation: the ONLY package of habitus that reads attack labels.

    labels      ground-truth node ids (HostSpec.gt_files under settings.paths.ground_truth), y over a universe
    metrics     oracle best-MCC confusion, AP / AUC over the universe, budget rows
    tables      the E1 / E2 ablation table (ablation_e1_e2.csv / .md, main_config.csv)
    derived     paper means and margins from the shipped results/ CSVs
    operating   budget rows, lambda / combiner sensitivity, per-form ablation (forms replay)
    bootstrap   paired stratified entity bootstrap of the oracle node MCC
    baselines   PIDSMaker baseline rows from the upstream run artifacts (server only)

The label guard (habitus.guard) blocks importing this package from every guarded command; label-free
inference that eval needs (the forms replay) runs as a guarded subprocess. Submodules are imported
explicitly; this file imports nothing.
"""

__all__ = ["labels", "metrics", "tables", "derived", "operating", "bootstrap", "baselines"]
