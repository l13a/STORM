r"""
Gene-program (GP) activity analysis for trained STORM models.

This module packages the post-training GP analysis pipeline used in the
STORM manuscript. The training-time GP *mask* construction lives in
:mod:`storm.preprocessing.gene_programs`; this module is for what you do
*after* a model has been fit:

1. **Project cells into the GP-activity latent space.** Each active gene
   program is a coordinate in
   ``adata.obsm[gp_key]`` (default ``"X_storm_gp"``) — the per-cell
   activity score of that program. Use :func:`encode_gp_activity`.
2. **Test which GPs are enriched in which clusters.** Per-cluster
   Wilcoxon rank-sum (vs. the rest) with FDR correction; see
   :func:`gp_enrichment_test` and :func:`top_gps_per_cluster`.
3. **Visualise.** Cluster × GP heatmaps (:func:`gp_cluster_heatmap`),
   target-gene weight composition bars (:func:`plot_gp_target_weights`),
   non-zero gene/peak count bars (:func:`plot_gp_nonzero_counts`), and
   the per-GP × per-timepoint spatial grid that is the canonical STORM
   temporal panel (:func:`plot_spatial_gp_activity_grid`).

All AnnData keys follow the ``X_storm*`` convention. Override via the
``gp_key`` / ``signc_key`` keyword arguments if your project uses
different names.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.figure import Figure
from scipy.sparse import issparse
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Step 1 — project cells into the GP-activity space
# ---------------------------------------------------------------------------

def encode_gp_activity(
        storm_model,
        key: str,
        adata: AnnData,
        graph,
        *,
        adata_rna: Optional[AnnData] = None,
        sign_adjusted: bool = True,
        only_active_gps: bool = True,
        batch_size: int = 128,
        gp_key: str = "X_storm_gp",
        signc_key: str = "X_storm_gp_signc",
) -> np.ndarray:
    r"""Run ``model.encode_gp_latent`` and store the result in ``adata.obsm``.

    Parameters
    ----------
    storm_model
        Trained :class:`storm.models.STORMModel` or
        :class:`storm.models.PairedSTORMModel`.
    key
        Modality key — typically ``"rna"`` or ``"atac"``.
    adata
        AnnData to encode. Must be the same modality as ``key`` and
        configured the same way the model was trained on
        (matching :func:`storm.models.configure_dataset` settings).
    graph
        The (HVF-subsetted) guidance graph used for training.
    adata_rna
        When ``key="atac"``, pass the matching RNA AnnData so the
        encoder can resolve the cross-modality references.
    sign_adjusted
        If ``True`` (recommended), STORM also writes a *sign-corrected*
        copy of the GP activity to ``adata.obsm[signc_key]``. Sign
        correction flips columns whose target-gene weights were
        predominantly negative so the resulting activity score correlates
        with elevated expression of the GP's anchor genes.
    only_active_gps
        Drop columns corresponding to inactive GPs (per
        ``model.get_active_gp_mask()``).
    batch_size
        Mini-batch size for encoding.
    gp_key
        ``obsm`` key under which to store the raw GP activity matrix.
    signc_key
        ``obsm`` key for the sign-adjusted matrix (only written if
        ``sign_adjusted=True``).

    Returns
    -------
    activity
        Shape ``(n_cells, n_active_gps)``. Also written to
        ``adata.obsm[gp_key]``; the sign-adjusted copy goes to
        ``adata.obsm[signc_key]``.
    """
    extra = {"adata_rna": adata_rna} if key == "atac" and adata_rna is not None else {}
    if sign_adjusted:
        activity, activity_signc, majority_sign = storm_model.encode_gp_latent(
            key, adata, graph,
            batch_size=batch_size,
            only_active_gps=only_active_gps,
            sign_adjusted=True,
            **extra,
        )
        adata.obsm[gp_key] = np.asarray(activity)
        adata.obsm[signc_key] = np.asarray(activity_signc)
    else:
        activity = storm_model.encode_gp_latent(
            key, adata, graph,
            batch_size=batch_size,
            only_active_gps=only_active_gps,
            sign_adjusted=False,
            **extra,
        )
        adata.obsm[gp_key] = np.asarray(activity)
    return adata.obsm[gp_key]


# ---------------------------------------------------------------------------
# Step 2 — per-cluster enrichment
# ---------------------------------------------------------------------------

def gp_enrichment_test(
        adata: AnnData,
        *,
        gp_summary: pd.DataFrame,
        cluster_key: str = "domain",
        gp_key: str = "X_storm_gp",
        fdr_method: str = "fdr_bh",
        zscore: bool = True,
) -> pd.DataFrame:
    r"""Wilcoxon rank-sum test of each GP against each cluster (one-vs-rest).

    For every cluster ``c`` in ``adata.obs[cluster_key]`` and every GP
    column in ``adata.obsm[gp_key]``, compute the rank-sum statistic of
    cells in ``c`` vs. cells not in ``c`` on that GP's activity, then
    apply FDR correction across the cluster axis.

    Parameters
    ----------
    adata
        AnnData with both ``adata.obsm[gp_key]`` and
        ``adata.obs[cluster_key]`` set.
    gp_summary
        DataFrame from ``model.get_gp_summary()`` /
        ``model.get_gp_summary_atac()``. Must have at least one row per
        column in ``adata.obsm[gp_key]``; column ``"gp_name"`` is read
        for human-readable labels.
    cluster_key
        Column in ``adata.obs`` holding the cluster labels.
    gp_key
        Key of the GP activity matrix in ``adata.obsm``.
    fdr_method
        Multi-testing method passed to
        :func:`statsmodels.stats.multitest.multipletests`. Default
        Benjamini-Hochberg (``"fdr_bh"``).
    zscore
        If ``True`` (default), z-score-normalise each GP column before
        the test — matches the convention in the manuscript.

    Returns
    -------
    df
        Long-form ``DataFrame`` with one row per ``(gp, cluster)`` pair
        and columns
        ``["gp_idx", "gp_name", "cluster", "rank_sum_stat", "p_value",
        "corrected_p_value", "abs_stat"]``.
    """
    matrix = np.asarray(adata.obsm[gp_key])
    labels = adata.obs[cluster_key].astype(str).values
    clusters = np.unique(labels)
    n_gps = matrix.shape[1]
    if len(gp_summary) < n_gps:
        raise ValueError(
            f"gp_summary has {len(gp_summary)} rows but the GP matrix has "
            f"{n_gps} columns. Pass the modality-specific GP summary "
            "(model.get_gp_summary() for rna, get_gp_summary_atac() for atac)."
        )

    rows = []
    for gp_idx in range(n_gps):
        expr = matrix[:, gp_idx].astype(float)
        if zscore:
            sd = expr.std()
            expr = (expr - expr.mean()) / sd if sd > 0 else expr - expr.mean()

        stats: List[float] = []
        pvals: List[float] = []
        for cl in clusters:
            mask = labels == cl
            stat, p = ranksums(expr[mask], expr[~mask])
            stats.append(stat)
            pvals.append(p)
        corrected = multipletests(pvals, method=fdr_method)[1]

        for cl, stat, p, q in zip(clusters, stats, pvals, corrected):
            rows.append({
                "gp_idx": gp_idx,
                "gp_name": gp_summary["gp_name"].iloc[gp_idx],
                "cluster": cl,
                "rank_sum_stat": stat,
                "p_value": p,
                "corrected_p_value": q,
                "abs_stat": abs(stat),
            })
    return pd.DataFrame(rows)


def top_gps_per_cluster(
        enrichment_df: pd.DataFrame,
        n: int = 5,
        *,
        by: str = "rank_sum_stat",
        ascending: bool = False,
        fdr_threshold: Optional[float] = None,
) -> pd.DataFrame:
    r"""Select the top-``n`` GPs per cluster from an enrichment table.

    Parameters
    ----------
    enrichment_df
        Output of :func:`gp_enrichment_test`.
    n
        Number of GPs to return per cluster.
    by
        Sort column — typically ``"rank_sum_stat"`` (default, larger =
        more enriched), ``"abs_stat"`` (largest absolute effect), or
        ``"corrected_p_value"`` (smallest FDR; use with
        ``ascending=True``).
    ascending
        Sort direction.
    fdr_threshold
        If set, drop rows whose ``corrected_p_value`` exceeds this
        before ranking.

    Returns
    -------
    df
        Subset of ``enrichment_df`` containing at most ``n`` rows per
        cluster, ordered first by cluster, then by the selected
        statistic.
    """
    df = enrichment_df
    if fdr_threshold is not None:
        df = df[df["corrected_p_value"] <= fdr_threshold]
    out = (
        df.sort_values(by, ascending=ascending)
          .groupby("cluster", as_index=False, sort=False)
          .head(n)
          .sort_values(["cluster", by], ascending=[True, ascending])
          .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Step 3 — visualisation
# ---------------------------------------------------------------------------

def _ensure_list(x) -> List:
    """Coerce a value that may be a list, ndarray, or stringified list to a list."""
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith("[") and xs.endswith("]"):
            try:
                return list(ast.literal_eval(xs))
            except (ValueError, SyntaxError):
                pass
        return [t.strip().strip("'\"") for t in x.strip("[]").split(",") if t.strip()]
    return [x]


def _ensure_float_list(x) -> List[float]:
    """Like _ensure_list but coerces values to floats; tolerates `np.float32(...)` strings."""
    if isinstance(x, list):
        return [float(y) for y in x]
    if isinstance(x, (np.ndarray, tuple)):
        return [float(y) for y in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return [float(n) for n in nums]


def gp_cluster_heatmap(
        adata: AnnData,
        gp_indices: Sequence[int],
        gp_names: Optional[Sequence[str]] = None,
        *,
        cluster_key: str = "domain",
        gp_key: str = "X_storm_gp",
        cluster_order: Optional[Sequence] = None,
        scale: str = "minmax",
        row_cluster: bool = False,
        cmap: str = "viridis",
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
) -> Tuple[Figure, pd.DataFrame]:
    r"""Average-per-cluster scaled heatmap for a subset of gene programs.

    Parameters
    ----------
    adata
        AnnData with ``obsm[gp_key]`` and ``obs[cluster_key]`` set.
    gp_indices
        Iterable of column indices into ``adata.obsm[gp_key]`` (these
        are the GPs to plot — typically the output of
        :func:`top_gps_per_cluster`).
    gp_names
        Optional human-readable labels for each GP (used as column
        labels in the heatmap). Length must match ``gp_indices``.
    cluster_key, gp_key
        AnnData key names.
    cluster_order
        Order of rows (cluster IDs). Defaults to ``sorted`` of the
        unique cluster labels.
    scale
        ``"minmax"`` (default) scales each GP column to ``[0, 1]``
        across all cells before averaging; ``"zscore"`` z-scores each
        column; ``None`` leaves raw values.
    row_cluster
        If ``True``, render via :func:`seaborn.clustermap` with rows
        hierarchically reordered. Otherwise a plain heatmap.
    cmap, figsize, title
        Pass-through plot options.

    Returns
    -------
    fig, heatmap_df
        The Matplotlib :class:`Figure` and the underlying
        ``(n_clusters × n_gps)`` DataFrame.
    """
    matrix = np.asarray(adata.obsm[gp_key])[:, list(gp_indices)].astype(float)
    if scale == "minmax":
        mn = matrix.min(axis=0, keepdims=True)
        mx = matrix.max(axis=0, keepdims=True)
        matrix = np.divide(matrix - mn, mx - mn, where=(mx - mn) > 0,
                           out=np.zeros_like(matrix))
    elif scale == "zscore":
        mu = matrix.mean(axis=0, keepdims=True)
        sd = matrix.std(axis=0, keepdims=True)
        matrix = np.divide(matrix - mu, sd, where=sd > 0,
                           out=np.zeros_like(matrix))

    domain = adata.obs[cluster_key]
    if cluster_order is None:
        cluster_order = sorted(domain.dropna().unique(),
                               key=lambda x: (str(type(x)), x))
    rows = []
    for cl in cluster_order:
        mask = (domain == cl).to_numpy()
        rows.append(matrix[mask].mean(axis=0))
    rows = np.vstack(rows)

    if gp_names is None:
        gp_names = [str(i) for i in gp_indices]
    df = pd.DataFrame(rows, index=list(cluster_order), columns=list(gp_names))
    if figsize is None:
        figsize = (max(6.0, len(gp_indices) * 0.5 + 4),
                   max(4.0, len(cluster_order) * 0.4 + 4))

    if row_cluster:
        cg = sns.clustermap(
            df, method="average", metric="euclidean",
            row_cluster=True, col_cluster=False,
            cmap=cmap, linewidths=0, figsize=figsize,
            cbar_kws={"label": "Scaled GP score"},
        )
        cg.ax_heatmap.set_xlabel("GP")
        cg.ax_heatmap.set_ylabel(cluster_key)
        if title is not None:
            cg.ax_heatmap.set_title(title)
        return cg.figure, df
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap=cmap, linewidths=0, ax=ax,
                cbar_kws={"label": "Scaled GP score"})
    ax.set_xlabel("GP")
    ax.set_ylabel(cluster_key)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, df


def plot_gp_target_weights(
        gp_summary: pd.DataFrame,
        gp_dict: Mapping,
        cluster_order: Sequence,
        *,
        top_n_genes_per_gp: Optional[int] = None,
        figsize_per_bar: float = 0.35,
        weights_col: str = "gp_target_genes_weights",
        genes_col: str = "gp_target_genes",
) -> Tuple[Figure, pd.DataFrame]:
    r"""Stacked-bar 'composition' plot of target-gene weights for a panel of GPs.

    For each GP listed in ``gp_dict[cluster]`` (one entry per cluster in
    ``cluster_order``), draws one stacked bar whose segments are the
    GP's target-gene weights (absolute values, normalised to sum to 1).
    Segments are stacked in descending size so the dominant gene is
    always at the bottom.

    Parameters
    ----------
    gp_summary
        DataFrame from ``model.get_gp_summary()`` /
        ``get_gp_summary_atac()``. Must contain ``"gp_name"``,
        ``"all_gp_idx"`` and the two list-valued columns named by
        ``genes_col`` and ``weights_col`` (default
        ``"gp_target_genes"`` / ``"gp_target_genes_weights"``).
    gp_dict
        Mapping ``{cluster_id: [gp_idx, ...]}`` — typically derived
        from :func:`top_gps_per_cluster`.
    cluster_order
        Iterable of cluster IDs whose GP panels should appear in order
        on the x-axis.
    top_n_genes_per_gp
        If set, truncate each GP to its top-N weighted genes before
        normalising.
    figsize_per_bar
        Inches of horizontal space per GP bar.
    weights_col, genes_col
        Column names in ``gp_summary``.

    Returns
    -------
    fig, weight_df
        Matplotlib figure and a tidy DataFrame with one row per
        ``(cluster, gp_idx, gene, normalised_weight)``.
    """
    def _get_row(df: pd.DataFrame, gp_idx: int) -> pd.Series:
        if "all_gp_idx" in df.columns:
            hits = df[df["all_gp_idx"].astype(int) == int(gp_idx)]
            if len(hits) == 1:
                return hits.iloc[0]
        if gp_idx in df.index:
            return df.loc[gp_idx]
        if 0 <= int(gp_idx) < len(df):
            return df.iloc[int(gp_idx)]
        raise KeyError(f"No GP row for index {gp_idx!r}.")

    bars, rows = [], []
    for cl in cluster_order:
        for gp_idx in gp_dict.get(cl, []):
            row = _get_row(gp_summary, gp_idx)
            genes = _ensure_list(row.get(genes_col))
            weights = _ensure_float_list(row.get(weights_col))
            if not genes or not weights:
                continue
            abs_w = np.abs(np.asarray(weights, dtype=float))
            mask = abs_w > 0
            genes = [g for g, m in zip(genes, mask) if m]
            abs_w = abs_w[mask]
            if abs_w.size == 0:
                continue
            if top_n_genes_per_gp is not None and len(genes) > top_n_genes_per_gp:
                order = np.argsort(abs_w)[::-1][:top_n_genes_per_gp]
                genes = [genes[i] for i in order]
                abs_w = abs_w[order]
            norm = abs_w / abs_w.sum()
            label = str(row.get("gp_name", f"GP{gp_idx}"))
            order = np.argsort(norm)[::-1]
            bars.append((label, [(genes[i], norm[i]) for i in order]))
            for g, w in zip(genes, norm):
                rows.append({"cluster": cl, "gp_idx": int(gp_idx),
                             "gp_name": label, "gene": g, "weight": float(w)})

    if not bars:
        raise ValueError("No GPs with non-zero target-gene weights to plot.")

    fig, ax = plt.subplots(figsize=(max(4.0, figsize_per_bar * len(bars) + 2), 5))
    palette = sns.color_palette("tab20", n_colors=20)
    x = np.arange(len(bars))
    bottoms = np.zeros(len(bars))
    for k in range(max(len(b[1]) for b in bars)):
        heights = np.zeros(len(bars))
        labels = []
        for i, (_, segs) in enumerate(bars):
            if k < len(segs):
                heights[i] = segs[k][1]
                labels.append(segs[k][0])
            else:
                labels.append("")
        ax.bar(x, heights, bottom=bottoms,
               color=[palette[(i * 7 + k) % len(palette)] for i in range(len(bars))])
        # annotate large enough segments
        for i, lab in enumerate(labels):
            if heights[i] > 0.04:
                ax.text(x[i], bottoms[i] + heights[i] / 2, lab,
                        ha="center", va="center", fontsize=7)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], rotation=90, fontsize=8)
    ax.set_ylabel("Normalised target-gene weight")
    ax.set_xlabel("Gene program")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def plot_gp_nonzero_counts(
        gp_summary: pd.DataFrame,
        gp_dict: Mapping,
        cluster_order: Sequence,
        *,
        weights_col: str = "gp_target_genes_weights",
        color: str = "#4C78A8",
        figsize_per_bar: float = 0.35,
        drop_zeros: bool = False,
        annotate: bool = True,
) -> Tuple[Figure, pd.Series]:
    r"""Bar plot of non-zero target counts per GP (one bar per GP in ``gp_dict``).

    Mirrors :func:`plot_gp_target_weights` but height = number of
    non-zero weights instead of weight composition. Useful for sanity-
    checking that the GPs you've selected actually have a meaningful
    number of contributing features.

    Parameters
    ----------
    weights_col
        ``"gp_target_genes_weights"`` for RNA or
        ``"gp_target_peaks_weights"`` for ATAC.
    """
    def _get_row(df: pd.DataFrame, gp_idx: int) -> pd.Series:
        if "all_gp_idx" in df.columns:
            hits = df[df["all_gp_idx"].astype(int) == int(gp_idx)]
            if len(hits) == 1:
                return hits.iloc[0]
        if 0 <= int(gp_idx) < len(df):
            return df.iloc[int(gp_idx)]
        raise KeyError(f"No GP row for index {gp_idx!r}.")

    labels, counts = [], []
    for cl in cluster_order:
        for gp_idx in gp_dict.get(cl, []):
            row = _get_row(gp_summary, gp_idx)
            weights = _ensure_float_list(row.get(weights_col))
            count = int(np.sum(np.abs(np.asarray(weights, dtype=float)) > 0)) if weights else 0
            if drop_zeros and count == 0:
                continue
            labels.append(str(row.get("gp_name", f"GP{gp_idx}")))
            counts.append(count)

    fig, ax = plt.subplots(figsize=(max(4.0, figsize_per_bar * len(labels) + 2), 4))
    x = np.arange(len(labels))
    ax.bar(x, counts, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Non-zero target features")
    ax.set_xlabel("Gene program")
    if annotate:
        for xi, c in zip(x, counts):
            ax.text(xi, c, str(c), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    return fig, pd.Series(counts, index=labels, name="non_zero_count")


def plot_spatial_gp_activity_grid(
        adata: AnnData,
        gp_indices: Sequence[int],
        timepoints: Sequence[str],
        *,
        gp_summary: Optional[pd.DataFrame] = None,
        gp_key: str = "X_storm_gp",
        timepoint_key: str = "timepoint",
        normalise: str = "zscore",
        bin_key: Optional[str] = None,
        spot_size: float = 1.2,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize_per_panel: Tuple[float, float] = (4.2, 4.6),
) -> Figure:
    r"""Per-GP × per-timepoint spatial grid of GP activity (the STORM temporal panel).

    Rows = timepoints, columns = GPs. Each panel is a spatial scatter
    coloured by the named GP's activity at that timepoint. This is the
    canonical figure used in the STORM manuscript to show how a
    program's spatial activity shifts across developmental time.

    Parameters
    ----------
    adata
        AnnData with ``obsm[gp_key]``, ``obsm["spatial"]``, and
        ``obs[timepoint_key]``.
    gp_indices
        Column indices into ``adata.obsm[gp_key]`` to plot.
    timepoints
        Timepoint IDs in the order they should appear top-to-bottom.
    gp_summary
        Optional — if provided, panel titles use the ``"gp_name"``
        column rather than the bare index.
    normalise
        Per-GP normalisation applied *before* slicing into timepoints
        so the colour scale is comparable across panels:
        ``"zscore"`` (default) → ``(x - mean) / std``;
        ``"minmax"`` → ``(x - min) / (max - min)``;
        ``"none"`` → raw values.
    bin_key
        Optional ``adata.obs`` column for spatial bin averaging. If
        set, the activity at each cell is replaced by the mean activity
        within its spatial bin (smoothing). Set to ``None`` to plot raw
        per-cell values.
    spot_size, cmap, vmin, vmax, figsize_per_panel
        Pass-through plotting controls.
    """
    matrix = np.asarray(adata.obsm[gp_key])
    n_rows, n_cols = len(timepoints), len(gp_indices)

    # Pre-compute the per-GP normaliser so all panels share a scale.
    if normalise == "zscore":
        mu = matrix[:, list(gp_indices)].mean(axis=0)
        sd = matrix[:, list(gp_indices)].std(axis=0)
        sd[sd <= 1e-12] = 1.0
        norm_fn = lambda v, j: (v - mu[j]) / sd[j]
        default_vmin, default_vmax = -2.0, 2.0
    elif normalise == "minmax":
        mn = matrix[:, list(gp_indices)].min(axis=0)
        mx = matrix[:, list(gp_indices)].max(axis=0)
        rng = mx - mn
        rng[rng <= 1e-12] = 1.0
        norm_fn = lambda v, j: (v - mn[j]) / rng[j]
        default_vmin, default_vmax = 0.0, 1.0
    else:
        norm_fn = lambda v, j: v
        default_vmin = default_vmax = None
    if vmin is None: vmin = default_vmin
    if vmax is None: vmax = default_vmax

    def _gp_title(idx: int) -> str:
        if gp_summary is None or "gp_name" not in gp_summary.columns:
            return f"GP {idx}"
        try:
            return str(gp_summary.iloc[idx]["gp_name"])
        except Exception:
            return f"GP {idx}"

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, tp in enumerate(timepoints):
        sub = adata[adata.obs[timepoint_key] == tp].copy()
        for k, gp_idx in enumerate(gp_indices):
            col = sub.obsm[gp_key][:, gp_idx]
            vals = norm_fn(np.asarray(col), k)
            if bin_key is not None and bin_key in sub.obs:
                vals = (
                    pd.Series(vals)
                      .groupby(pd.Series(sub.obs[bin_key].to_numpy()))
                      .transform("mean")
                      .to_numpy()
                )
            obs_col = f"_storm_gp_{gp_idx}"
            sub.obs[obs_col] = vals
            sc.pl.spatial(
                sub, color=obs_col, spot_size=spot_size, cmap=cmap,
                vmin=vmin, vmax=vmax,
                title=_gp_title(gp_idx) if i == 0 else None,
                ax=axes[i, k], show=False,
                colorbar_loc="right" if i == n_rows - 1 else None,
            )
            if k == 0:
                axes[i, k].set_ylabel(tp, fontsize=11)
            else:
                axes[i, k].set_ylabel("")
            axes[i, k].invert_yaxis()
    fig.tight_layout()
    return fig
