#!/usr/bin/env python
"""Stage the 5-timepoint temporal-analysis inputs into the STORM tutorial tree.

Tutorial 6 demonstrates STORM's *temporal* gene-program analysis on a
5-timepoint (Q43P0, Q43P2, Q43P5, C8P10, Q43P21) postnatal mouse-brain dataset.
That model was trained on a *separate* cluster following the same tutorial 1-5
pipeline; only its STORMed outputs and a few intermediate tables are available
here (there is no ``.dill`` checkpoint, and none is needed — the per-cell
program activities are already baked into the saved h5ads).

This script reads those saved artifacts (READ-ONLY) from ``temp_data/``,
restricts them to the corpus-callosum spatial-bin subset used for the
trajectory analysis, attaches the spatial-bin (``final_bin``) and
developmental-time (``time``) labels, renames the GP-activity ``obsm`` keys to
the ``storm_*`` convention, and writes analysis-ready files under
``artifacts/storm_tutorial_5tmp/``.

It does NOT modify anything under ``GLUE_GP/`` or ``temp_data/``. The heavy
(2.9 GB) gene-activity-score file is intentionally not staged — the temporal
trajectory analysis runs entirely off the saved program activities.
"""
import os
import re
import shutil

import anndata as ad
import pandas as pd

DATA_ROOT = "/gpfs/gibbs/pi/zhao/xc384/data"
VIZ = f"{DATA_ROOT}/temp_data/viz_GP_5tmp"
EXPORT = f"{DATA_ROOT}/temp_data/5tmp_c17_export"

OUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir,
                 "artifacts", "storm_tutorial_5tmp"))
os.makedirs(OUT, exist_ok=True)


def normalize_barcode(barcode):
    """Match the training-run barcode convention: ensure a trailing ``-1``."""
    return re.sub(r"-\d$", "-1", barcode) if re.search(r"-\d$", barcode) else barcode + "-1"


def load_subset_rekey(path, cell_names, metadata):
    a = ad.read_h5ad(path)
    # Reconstruct the `{timepoint}#{barcode}-1` names the metadata is keyed on.
    a.obs_names = [f"{tp}#{normalize_barcode(bc)}"
                   for bc, tp in zip(a.obs_names, a.obs["timepoint"])]
    a = a[cell_names].copy()                       # corpus-callosum subset
    a.obs["final_bin"] = (
        metadata["final_bin"].reindex(a.obs_names).astype(int).values)
    a.obs["time"] = ["P" + s.split("P")[1] for s in a.obs["timepoint"]]
    # Adopt the storm_* obsm convention used by tutorials 1-5.
    for old, new in [("X_glue", "X_storm"),
                     ("X_glue_gp", "X_storm_gp"),
                     ("X_glue_gp_signc", "X_storm_gp_signc")]:
        if old in a.obsm:
            a.obsm[new] = a.obsm[old]
            del a.obsm[old]
    return a


def main():
    cell_names = list(pd.read_csv(f"{EXPORT}/cell_names_CC_nobin12.csv")["x"])
    metadata = pd.read_csv(f"{EXPORT}/CC_metadata_nobin12.csv", index_col=0)

    rna = load_subset_rekey(f"{VIZ}/rna_5tmp_GLUPED.h5ad", cell_names, metadata)
    atac = load_subset_rekey(f"{VIZ}/atac_5tmp_GLUPED.h5ad", cell_names, metadata)
    rna.write(f"{OUT}/rna_5tmp_cc.h5ad", compression="gzip")
    atac.write(f"{OUT}/atac_5tmp_cc.h5ad", compression="gzip")

    # GP summaries, temporal module labels, and precomputed time x bin
    # trajectory matrices (all small, copied verbatim with clearer names).
    shutil.copy2(f"{VIZ}/subset.csv", f"{OUT}/gp_summary_rna.csv")
    shutil.copy2(f"{VIZ}/subset_atac.csv", f"{OUT}/gp_summary_atac.csv")
    shutil.copy2(f"{EXPORT}/gps_joint_cluster_labels_CC_corred_nobin12.csv",
                 f"{OUT}/gp_temporal_modules.csv")
    shutil.copy2(f"{EXPORT}/rna_agg_df_mat_corred_nobin12.csv",
                 f"{OUT}/rna_module_trajectories.csv")
    shutil.copy2(f"{EXPORT}/atac_agg_df_mat_corred_nobin12.csv",
                 f"{OUT}/atac_module_trajectories.csv")

    print(f"RNA  CC subset -> {OUT}/rna_5tmp_cc.h5ad  ({rna.n_obs} x {rna.n_vars})")
    print(f"ATAC CC subset -> {OUT}/atac_5tmp_cc.h5ad ({atac.n_obs} x {atac.n_vars})")
    print("timepoints:", sorted(rna.obs["timepoint"].astype(str).unique()))
    print("final_bin :", int(rna.obs["final_bin"].min()), "-",
          int(rna.obs["final_bin"].max()))
    print("Also copied: gp_summary_rna/atac.csv, gp_temporal_modules.csv, "
          "rna/atac_module_trajectories.csv")


if __name__ == "__main__":
    main()
