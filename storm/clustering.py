r"""
Clustering helpers for the STORM joint latent space.

Two layers:

* **Generic helpers** — :func:`search_res`, :func:`search_res_min_geq`,
  :func:`merge_to_target`, :func:`mclust_R`, :func:`refine_label`,
  :func:`cluster`. These operate on any AnnData with an embedding in
  ``adata.obsm`` and write a ``"domain"`` column into ``adata.obs``.
* **Paired-data wrappers** — :func:`concat_clustering`,
  :func:`joint_clustering`, :func:`joint_clustering_auc_select`. These
  bundle the recurring "RNA + ATAC are paired, do something with both"
  recipes used by the STORM tutorials and reproduction scripts.

The CONCAT vs. JOINT distinction (see :func:`concat_clustering` /
:func:`joint_clustering`) is documented in
``examples/tutorial_3_clustering.py``; in short, CONCAT row-stacks the two
modalities and clusters once (each cell gets its own label), while JOINT
element-wise-adds the per-modality latents (one label per spatial location).
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Generic resolution / merge helpers
# ---------------------------------------------------------------------------

def search_res(
        adata: AnnData,
        n_clusters: int,
        method: str = "leiden",
        use_rep: str = "X_storm",
        start: float = 0.1,
        end: float = 3.0,
        increment: float = 0.01,
        n_neighbors: int = 50,
        random_state: int = 0,
) -> float:
    r"""Search the largest resolution that yields exactly ``n_clusters``.

    Walks resolutions from ``end`` down to ``start`` in steps of
    ``increment`` and returns the first one that produces exactly
    ``n_clusters`` clusters under the given ``method``
    (``"leiden"`` | ``"louvain"``).

    Raises
    ------
    AssertionError
        If no resolution in the range yields the exact target count.
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    label = 0
    res = None
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=random_state, resolution=res)
            count = adata.obs["leiden"].nunique()
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=random_state, resolution=res)
            count = adata.obs["louvain"].nunique()
        else:
            raise ValueError(f"Unknown method: {method!r}")
        if count == n_clusters:
            label = 1
            break
    assert label == 1, (
        "Resolution not found. Please try a bigger range or smaller step."
    )
    return res


def search_res_min_geq(
        adata: AnnData,
        min_clusters: int,
        method: str = "leiden",
        use_rep: str = "X_storm",
        start: float = 0.1,
        end: float = 3.0,
        step: float = 0.05,
        n_neighbors: int = 30,
        random_state: int = 0,
) -> float:
    r"""Smallest resolution that produces at least ``min_clusters`` clusters.

    Used as the first step of the louvain-merge / leiden-merge recipe in
    :func:`cluster`: over-cluster slightly, then agglomeratively merge the
    centroids back down to the target count.
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    for res in np.arange(start, end + 1e-9, step):
        if method == "leiden":
            sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added="tmp")
        else:
            sc.tl.louvain(adata, resolution=res, random_state=random_state, key_added="tmp")
        if adata.obs["tmp"].nunique() >= min_clusters:
            adata.obs.drop(columns=["tmp"], inplace=True)
            return res
    raise RuntimeError(
        f"Could not reach {min_clusters} clusters within [{start}, {end}]."
    )


def merge_to_target(
        adata: AnnData,
        label_key: str,
        target_n: int,
        rep_key: str = "X_storm",
        linkage: str = "ward",
) -> pd.Series:
    r"""Agglomeratively merge a fine-grained labelling down to ``target_n`` clusters.

    Uses Euclidean distance between cluster centroids in
    ``adata.obsm[rep_key]``. Returns a pandas ``Series`` (indexed like
    ``adata.obs``) mapping each cell to its merged integer label.
    """
    groups = adata.obs.groupby(label_key).indices
    old_labels: List = []
    centroids: List[np.ndarray] = []
    for lab, idx in groups.items():
        old_labels.append(lab)
        centroids.append(adata.obsm[rep_key][idx].mean(axis=0))
    centroids = np.vstack(centroids)
    agg = AgglomerativeClustering(
        n_clusters=target_n, metric="euclidean", linkage=linkage,
    )
    new_ids = agg.fit_predict(centroids)
    mapping = {old: int(new) for old, new in zip(old_labels, new_ids)}
    return adata.obs[label_key].map(mapping).astype("category")


# ---------------------------------------------------------------------------
# mclust (R) backend
# ---------------------------------------------------------------------------

def mclust_R(
        adata: AnnData,
        num_cluster: int,
        modelNames: str = "EEE",
        used_obsm: str = "emb_pca",
        random_seed: int = 2020,
) -> AnnData:
    r"""Gaussian-mixture clustering via the R ``mclust`` package.

    Requires ``rpy2`` and an R installation with the ``mclust`` package.
    Writes integer-categorical labels into ``adata.obs['mclust']``.

    Parameters
    ----------
    adata
        AnnData with the embedding to cluster in ``adata.obsm[used_obsm]``.
    num_cluster
        Target number of mixture components (``G`` argument to ``Mclust``).
    modelNames
        Covariance-structure code passed to ``Mclust`` (default ``"EEE"``).
    used_obsm
        Key of the embedding in ``adata.obsm`` (default ``"emb_pca"`` to
        match the convention used by :func:`cluster`).
    random_seed
        Seed for both NumPy and R's RNG.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(
        rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]),
        num_cluster, modelNames,
    )
    mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")
    return adata


# ---------------------------------------------------------------------------
# Generic clustering entry point
# ---------------------------------------------------------------------------

def cluster(
        adata: AnnData,
        n_clusters: int = 14,
        key: str = "X_storm",
        method: str = "louvain_merge",
        start: float = 0.1,
        end: float = 3.0,
        increment: float = 0.05,
        overshoot: float = 1.3,
        n_neighbors: int = 30,
        random_state: int = 0,
        pca: bool = False,
        pca_n_components: int = 32,
        refinement: bool = False,
        radius: int = 5,
        mclust_model: str = "EEE",
) -> AnnData:
    r"""Cluster cells on a learned representation and write to ``adata.obs['domain']``.

    Mirrors ``bench_range.clustering``. The ``"mclust"`` branch requires R
    via ``rpy2`` and an installed ``mclust`` R package; the ``_merge``
    variants are the over-cluster-then-merge recipe used in the STORM
    paper.

    Parameters
    ----------
    adata
        AnnData with the embedding in ``adata.obsm[key]``.
    n_clusters
        Target number of clusters.
    key
        Key of the embedding in ``adata.obsm`` (default ``"X_storm"``).
    method
        One of ``{"mclust", "leiden", "louvain", "leiden_merge",
        "louvain_merge"}``. The ``_merge`` variants over-cluster by
        ``overshoot`` then agglomeratively merge centroids down to
        ``n_clusters``. ``"mclust"`` calls into R via :func:`mclust_R`.
    start, end, increment
        Resolution sweep range used by the underlying searches.
    overshoot
        For ``*_merge`` methods, over-cluster to
        ``ceil(n_clusters * overshoot)`` before merging.
    n_neighbors, random_state
        Passed to :func:`scanpy.pp.neighbors` / :func:`scanpy.tl.louvain`.
    pca
        If ``True``, run PCA to ``pca_n_components`` first and cluster on
        the resulting space (stored as ``adata.obsm['emb_pca']``). Default
        ``False`` — typically you want to cluster on ``X_storm`` directly.
    refinement
        If ``True``, run :func:`refine_label` to spatially smooth the
        ``domain`` labels (requires ``adata.obsm['spatial']`` and
        ``adata.obs['modality']`` / ``adata.obs['timepoint']``).
    radius
        Neighbour radius for the optional spatial refinement.
    mclust_model
        Covariance-structure code passed through to :func:`mclust_R` when
        ``method="mclust"`` (default ``"EEE"``).

    Returns
    -------
    adata
        The input ``AnnData`` (modified in place — ``adata.obs['domain']``
        is written, plus ``adata.obs['domain_high']`` for ``*_merge``
        methods).
    """
    if pca:
        embedding = PCA(
            n_components=pca_n_components, random_state=42,
        ).fit_transform(adata.obsm[key].copy())
        adata.obsm["emb_pca"] = embedding
    else:
        adata.obsm["emb_pca"] = adata.obsm[key].copy()

    if method == "mclust":
        mclust_R(
            adata, num_cluster=n_clusters, modelNames=mclust_model,
            used_obsm="emb_pca", random_seed=random_state,
        )
        adata.obs["domain"] = adata.obs["mclust"]
    elif method == "leiden":
        res = search_res(
            adata, n_clusters, method="leiden", use_rep="emb_pca",
            start=start, end=end, increment=increment,
            n_neighbors=n_neighbors, random_state=random_state,
        )
        sc.tl.leiden(adata, random_state=random_state, resolution=res)
        adata.obs["domain"] = adata.obs["leiden"]
    elif method == "louvain":
        res = search_res(
            adata, n_clusters, method="louvain", use_rep="emb_pca",
            start=start, end=end, increment=increment,
            n_neighbors=n_neighbors, random_state=random_state,
        )
        sc.tl.louvain(adata, random_state=random_state, resolution=res)
        adata.obs["domain"] = adata.obs["louvain"]
    elif method in {"leiden_merge", "louvain_merge"}:
        base = method.split("_")[0]
        high_k = max(n_clusters + 1, int(np.ceil(n_clusters * overshoot)))
        res = search_res_min_geq(
            adata, high_k, method=base, use_rep="emb_pca",
            start=start, end=end, step=increment,
            n_neighbors=n_neighbors, random_state=random_state,
        )
        if base == "leiden":
            sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added="domain_high")
        else:
            sc.tl.louvain(adata, resolution=res, random_state=random_state, key_added="domain_high")
        adata.obs["domain"] = merge_to_target(
            adata, "domain_high", target_n=n_clusters, rep_key="emb_pca",
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    adata.obs["domain"] = adata.obs["domain"].astype(int).astype("category")
    if refinement:
        refine_label(adata, radius=radius, key="domain")
    return adata


# ---------------------------------------------------------------------------
# Optional spatial label refinement
# ---------------------------------------------------------------------------

def refine_label(
        adata: AnnData,
        radius: int = 5,
        key: str = "domain",
        spatial_key: str = "spatial",
) -> AnnData:
    r"""Majority-vote label smoothing over each cell's ``radius`` spatial neighbours.

    Operates per ``(modality, timepoint)`` slice so cross-modality and
    cross-timepoint neighbours do not bleed into each other. Requires
    ``adata.obs['modality']`` and ``adata.obs['timepoint']`` to be set.
    """
    import ot  # POT — lazy because it's a heavy dep
    n_neigh = radius
    timepoints = adata.obs["timepoint"].unique()
    modalities = adata.obs["modality"].unique()

    for modality in modalities:
        for time in timepoints:
            sub = adata[
                (adata.obs["modality"] == modality)
                & (adata.obs["timepoint"] == time)
            ].copy()
            if sub.n_obs == 0:
                continue
            old = sub.obs[key].values
            distance = ot.dist(sub.obsm[spatial_key], sub.obsm[spatial_key],
                               metric="euclidean")
            new = []
            for i in range(distance.shape[0]):
                order = distance[i, :].argsort()
                neigh = [old[order[j]] for j in range(1, n_neigh + 1)]
                new.append(max(neigh, key=neigh.count))
            mask = (adata.obs["modality"] == modality) & (adata.obs["timepoint"] == time)
            adata.obs.loc[mask, key] = new
    return adata


# ---------------------------------------------------------------------------
# Paired-data wrappers — the three clustering modes used in the STORM paper
# ---------------------------------------------------------------------------

def _check_paired(rna: AnnData, atac: AnnData, key: str) -> None:
    if rna.n_obs != atac.n_obs:
        raise ValueError(
            f"RNA and ATAC must have the same number of paired cells; got "
            f"{rna.n_obs} and {atac.n_obs}."
        )
    if key not in rna.obsm or key not in atac.obsm:
        raise KeyError(
            f"Both AnnDatas must have an embedding at adata.obsm[{key!r}]."
        )


def concat_clustering(
        rna: AnnData, atac: AnnData,
        n_clusters: int,
        key: str = "X_storm",
        method: str = "louvain_merge",
        **kwargs,
) -> Tuple[AnnData, AnnData]:
    r"""CONCAT clustering: row-stack RNA + ATAC, cluster once, split labels.

    Mode A from ``examples/tutorial_3_clustering.py``. Returns the input
    AnnDatas (modified in place — ``rna.obs['domain']`` and
    ``atac.obs['domain']`` are written). RNA and ATAC end up with cluster
    labels in the **same label space** (each label ID means the same
    thing on both maps), but each cell's assignment is derived from its
    own row in the stacked embedding — *its own omics profile*.

    Use this to ask "do RNA and ATAC tell consistent spatial-domain
    stories?" — plot the same colour map on each modality's spatial
    layout and compare.

    Parameters
    ----------
    rna, atac
        Per-modality AnnDatas. Must be paired (same length, same row
        order — RNA's cell *i* and ATAC's cell *i* are at the same spot).
    n_clusters
        Target number of clusters.
    key
        Embedding key in both ``rna.obsm`` and ``atac.obsm`` (default
        ``"X_storm"``).
    method
        Clustering method — see :func:`cluster`.
    **kwargs
        Forwarded to :func:`cluster` (e.g. ``overshoot``, ``n_neighbors``,
        ``random_state``).

    Returns
    -------
    rna, atac
        The two input AnnDatas with ``obs['domain']`` filled in.
    """
    _check_paired(rna, atac, key)
    rna.obs["modality"] = "rna"
    atac.obs["modality"] = "atac"
    combined = ad.concat([rna, atac])
    combined.obsm[key] = np.vstack([rna.obsm[key], atac.obsm[key]])

    cluster(combined, n_clusters=n_clusters, key=key, method=method, **kwargs)

    split = rna.n_obs
    rna.obs["domain"] = combined.obs["domain"].iloc[:split].values
    atac.obs["domain"] = combined.obs["domain"].iloc[split:].values
    rna.obs["domain"] = rna.obs["domain"].astype(int).astype("category")
    atac.obs["domain"] = atac.obs["domain"].astype(int).astype("category")
    return rna, atac


def joint_clustering(
        rna: AnnData, atac: AnnData,
        n_clusters: int,
        key: str = "X_storm",
        method: str = "louvain_merge",
        template: str = "rna",
        **kwargs,
) -> AnnData:
    r"""JOINT clustering: element-wise add the two embeddings, cluster once.

    Mode B from ``examples/tutorial_3_clustering.py``. Returns a *new*
    AnnData with one row per spatial location whose ``obsm[key]`` is
    ``rna.obsm[key] + atac.obsm[key]`` and ``obs['domain']`` is the
    JOINT cluster labelling. ``obs`` and ``obsm`` are otherwise copied
    from ``template`` (``"rna"`` or ``"atac"``).

    Use this to assign a single canonical spatial-domain label per
    location based on the **combined** RNA + ATAC profile.

    Parameters
    ----------
    rna, atac
        Per-modality, paired AnnDatas.
    n_clusters
        Target number of clusters.
    key
        Embedding key in ``adata.obsm`` (default ``"X_storm"``).
    method
        Clustering method — see :func:`cluster`.
    template
        Which input to base the returned AnnData's ``obs`` / ``obsm`` /
        ``var`` on. ``"rna"`` (default) or ``"atac"``.
    **kwargs
        Forwarded to :func:`cluster`.

    Returns
    -------
    joint
        New AnnData with ``obsm[key] = rna + atac`` and
        ``obs['domain']`` set.
    """
    _check_paired(rna, atac, key)
    src = rna if template == "rna" else atac
    joint = src.copy()
    joint.obsm[key] = rna.obsm[key] + atac.obsm[key]
    cluster(joint, n_clusters=n_clusters, key=key, method=method, **kwargs)
    return joint


def _best_auc_for_structure(
        domain_labels: pd.Series,
        expression: np.ndarray,
        var_names,
        markers: Iterable[str],
) -> float:
    """Highest single-cluster AUC across ``domain_labels`` for one marker set."""
    var_map = {v.lower(): v for v in var_names}
    resolved = [var_map[g.lower()] for g in markers if g.lower() in var_map]
    if not resolved:
        return float("nan")
    cols = [list(var_names).index(g) for g in resolved]
    expr_mat = expression[:, cols]
    expr = np.asarray(expr_mat.mean(axis=1)).ravel()
    aucs = []
    for cl in domain_labels.cat.categories:
        mask = (domain_labels == cl).to_numpy()
        if mask.sum() == 0 or mask.sum() == len(mask):
            continue
        aucs.append(roc_auc_score(mask.astype(int), expr))
    return float(max(aucs)) if aucs else float("nan")


def joint_clustering_auc_select(
        rna: AnnData, atac: AnnData,
        n_clusters_sweep: Iterable[int],
        marker_dict: Mapping[str, Iterable[str]],
        key: str = "X_storm",
        method: str = "louvain_merge",
        scoring_layer: Optional[str] = "counts",
        template: str = "rna",
        **kwargs,
) -> Tuple[AnnData, pd.DataFrame, int]:
    r"""JOINT clustering swept over ``n_clusters_sweep``, AUC-selecting the best.

    For each candidate ``n``, runs :func:`joint_clustering` and scores it
    by asking, for each ``(structure → markers)`` entry in
    ``marker_dict``: "can a single cluster be separated from the rest
    using only the mean expression of these markers?" The best per-
    structure AUC is averaged across structures; the ``n`` maximising
    that average is returned.

    Parameters
    ----------
    rna, atac
        Per-modality, paired AnnDatas (see :func:`joint_clustering`).
    n_clusters_sweep
        Iterable of candidate cluster counts to try.
    marker_dict
        Mapping from anatomical-structure name → list of canonical RNA
        marker gene symbols (case-insensitive).
    key
        Embedding key in ``adata.obsm``.
    method
        Clustering method — see :func:`cluster`.
    scoring_layer
        Name of ``rna.layers`` to use as the expression matrix for AUC
        scoring. ``None`` uses ``rna.X``. Default ``"counts"``.
    template
        See :func:`joint_clustering`.
    **kwargs
        Forwarded to :func:`cluster`.

    Returns
    -------
    best_joint
        JOINT-clustered AnnData at the AUC-selected ``n_clusters``.
    auc_df
        Long-form DataFrame with columns
        ``["n_clusters", "structure", "best_auc"]`` for every
        ``(n, structure)`` pair scored.
    best_n
        The selected cluster count.
    """
    counts = rna.layers[scoring_layer] if scoring_layer is not None else rna.X
    counts_dense = counts.toarray() if hasattr(counts, "toarray") else counts

    rows = []
    by_n: dict = {}
    for n in n_clusters_sweep:
        joint_n = joint_clustering(
            rna, atac, n_clusters=n, key=key, method=method,
            template=template, **kwargs,
        )
        by_n[n] = joint_n
        for structure, markers in marker_dict.items():
            auc = _best_auc_for_structure(
                joint_n.obs["domain"], counts_dense, rna.var_names, markers,
            )
            rows.append({"n_clusters": n, "structure": structure, "best_auc": auc})

    auc_df = pd.DataFrame(rows)
    avg = auc_df.groupby("n_clusters")["best_auc"].mean()
    best_n = int(avg.idxmax())
    return by_n[best_n], auc_df, best_n
