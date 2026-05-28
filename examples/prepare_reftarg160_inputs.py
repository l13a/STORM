#!/usr/bin/env python
"""Reconcile the reftarg160 checkpoint's training preprocessing into the
standard STORM tutorial paths.

Why this exists
---------------
``dill_files/FINE_storm_P21P22_reftarg160_2tmp_RNAr_peak_source_repro.dill`` was
converted from a scglue-era model that was trained on a preprocessing producing
**2368** gene programs. Tutorial 1, run later, rebuilds its gene-program
dictionary from prior-knowledge databases that are downloaded live
(``load_from_disk=False``) and have since drifted, so it emits a *different*
program set (**2380**). Pairing that h5ad with this checkpoint makes
``encode_gp_latent`` raise an ``IndexError`` (2380 program names vs a 2368-wide
activity matrix).

The cells, genes and peaks are identical (same order) between the two
preprocessings; only the gene-program dictionary and the AnnData key prefix
differ (``nichecompass_*`` in the original vs ``storm_*`` after the refactor).
This script reads the checkpoint's own training preprocessing (READ-ONLY) and
writes ``storm``-keyed copies to the standard tutorial paths so tutorials 3-5
line up with the checkpoint. The original tutorial-1 outputs are backed up
first, so this is reversible.

It does NOT modify anything under ``GLUE_GP/`` or ``temp_data/``.
"""
import os
import shutil

import anndata as ad

# --- source: the checkpoint's own training preprocessing (READ-ONLY) -------
DATA_ROOT = "/gpfs/gibbs/pi/zhao/xc384/data"
SRC_RNA = f"{DATA_ROOT}/temp_data/rna-reftarg160_2tmp_prev_RNAr_peak_repro.h5ad"
SRC_ATAC = f"{DATA_ROOT}/temp_data/atac-reftarg160_2tmp_prev_RNAr_peak_repro.h5ad"
SRC_GRAPH = f"{DATA_ROOT}/temp_data/guidance.graphml160-2tmp_prev_RNAr_peak_repro.gz"

# --- destination: the standard tutorial paths ------------------------------
PREP_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir,
                 "artifacts", "storm_tutorial", "preprocessed"))
DST_RNA = f"{PREP_DIR}/rna_preprocessed.h5ad"
DST_ATAC = f"{PREP_DIR}/atac_preprocessed.h5ad"
DST_GRAPH = f"{PREP_DIR}/guidance.graphml.gz"
BACKUP_DIR = f"{PREP_DIR}/_backup_tutorial1_2380gp"

# Ground-truth cell-type label columns that tutorial 4 reads but that the
# original training h5ad does not carry; copied over from the backed-up files.
CARRY_OBS = ["RNA_clusters", "ATAC_clusters"]


def rekey(adata):
    """Rename ``nichecompass_*`` varm/uns keys to ``storm_*``; ensure HVG flag."""
    for store in (adata.varm, adata.uns):
        for key in list(store.keys()):
            if key.startswith("nichecompass_"):
                store["storm_" + key[len("nichecompass_"):]] = store[key]
                del store[key]
    if "highly_variable" not in adata.var:
        adata.var["highly_variable"] = True
    return adata


def carry_obs(adata, backup_path):
    """Copy ground-truth label columns from the backed-up tutorial-1 h5ad."""
    if not os.path.exists(backup_path):
        return adata
    old = ad.read_h5ad(backup_path, backed="r").obs
    for col in CARRY_OBS:
        if col in old.columns and col not in adata.obs.columns:
            adata.obs[col] = old.loc[adata.obs_names, col].values
    return adata


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # 1. Back up the current tutorial-1 outputs (reversible).
    for path in (DST_RNA, DST_ATAC, DST_GRAPH):
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(BACKUP_DIR, os.path.basename(path)))
    print(f"Backed up current preprocessing to {BACKUP_DIR}")

    # 2. Reconcile RNA / ATAC and write to the standard paths.
    rna = carry_obs(rekey(ad.read_h5ad(SRC_RNA)),
                    os.path.join(BACKUP_DIR, os.path.basename(DST_RNA)))
    atac = carry_obs(rekey(ad.read_h5ad(SRC_ATAC)),
                     os.path.join(BACKUP_DIR, os.path.basename(DST_ATAC)))

    rna.write(DST_RNA, compression="gzip")
    atac.write(DST_ATAC, compression="gzip")
    shutil.copy2(SRC_GRAPH, DST_GRAPH)

    n_gp = len(rna.uns["storm_gp_names"])
    print(f"RNA   -> {DST_RNA}  ({rna.n_obs} x {rna.n_vars}; {n_gp} gene programs)")
    print(f"ATAC  -> {DST_ATAC}  ({atac.n_obs} x {atac.n_vars})")
    print(f"graph -> {DST_GRAPH}")
    print("Done. Re-running tutorial 1 will overwrite these with a freshly "
          "downloaded (and likely drifted) program set; re-run this script "
          "afterwards to restore the checkpoint-matching inputs.")


if __name__ == "__main__":
    main()
