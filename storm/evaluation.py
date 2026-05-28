r"""
Evaluation metrics for trained STORM models.

This module packages the per-timepoint, AnnData-aware metric functions
that ``bench_range.py`` calls in the STORM reproduction pipeline. Every
function takes one or more AnnDatas plus key strings, loops over the
samples in ``adata.obs[sample_key]``, and returns a tidy dict
``{sample: score, "Overall": score}`` or — for the top-level harness
:func:`benchmark` — a tidy ``pandas.DataFrame``.

Four families of metrics:

* **Multi-omics integration** — :func:`foscttm_paired`, :func:`mlisi`,
  :func:`consistency_ari`.
* **Batch / timepoint integration** — :func:`pcr_score`, :func:`blisi`.
* **Cell-type recovery** — :func:`nmi_score`.
* **Niche coherence + spatial conservation** — :func:`lisi`,
  :func:`cell_type_asw`, :func:`map_score`, :func:`global_morans_i`,
  :func:`clisis`.

**Dependency note.** ``pcr_score`` and :func:`nmi_score` need
`scib <https://github.com/theislab/scib>`__; :func:`global_morans_i`
needs `squidpy <https://github.com/scverse/squidpy>`__. These are
imported lazily, so importing :mod:`storm.evaluation` works even without
them — a clear ``ImportError`` is raised only when you try to invoke a
metric that needs the missing dep.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_rand_score,
    pairwise_distances,
    silhouette_samples,
)
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _resolve_samples(adata: AnnData, samples: Optional[Iterable[str]],
                     sample_key: str) -> List[str]:
    if samples is not None:
        return list(samples)
    if sample_key not in adata.obs:
        return []
    return list(adata.obs[sample_key].astype(str).unique())


# ---------------------------------------------------------------------------
# Multi-omics integration
# ---------------------------------------------------------------------------

def foscttm_paired(
        rna: AnnData, atac: AnnData,
        embed_key: str = "X_storm",
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""FOSCTTM averaged across rna→atac, atac→rna; per-sample + overall.

    Returns ``1 - avg(FOSCTTM)`` so higher is better, matching the
    convention in the STORM paper tables. RNA and ATAC are assumed paired
    in row order.
    """
    def _one(src: np.ndarray, tgt: np.ndarray) -> float:
        d = cdist(src, tgt, "euclidean")
        return float(
            np.mean(np.sum(d < np.diag(d)[:, None], axis=1) / d.shape[1])
        )

    out: dict = {}
    a2b = _one(rna.obsm[embed_key], atac.obsm[embed_key])
    b2a = _one(atac.obsm[embed_key], rna.obsm[embed_key])
    out["Overall"] = round(1 - 0.5 * (a2b + b2a), ndigits)

    for tp in _resolve_samples(rna, samples, sample_key):
        sub_r = rna[rna.obs[sample_key] == tp].obsm[embed_key]
        sub_a = atac[atac.obs[sample_key] == tp].obsm[embed_key]
        a2b = _one(sub_r, sub_a)
        b2a = _one(sub_a, sub_r)
        out[tp] = round(1 - 0.5 * (a2b + b2a), ndigits)
    return out


def mlisi(
        rna: AnnData, atac: AnnData,
        embed_key: str = "X_storm",
        k: int = 15,
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""Modality-LISI: local mixing of RNA vs. ATAC neighbours, in [0, 1].

    For each cell, counts the modality composition of its ``k`` nearest
    neighbours in the joint embedding; returns the mean rescaled inverse
    Simpson index. Higher = better mixing.
    """
    def _one(rna_emb: np.ndarray, atac_emb: np.ndarray) -> float:
        X = np.concatenate([rna_emb, atac_emb], axis=0)
        batches = np.array(
            ["RNA"] * rna_emb.shape[0] + ["ATAC"] * atac_emb.shape[0]
        )
        D = pairwise_distances(X, metric="euclidean")
        scores = []
        for i in range(X.shape[0]):
            nn = batches[np.argsort(D[i])[:k]]
            _, counts = np.unique(nn, return_counts=True)
            scores.append(1.0 / np.sum((counts / k) ** 2))
        return float(np.mean((np.asarray(scores) - 1) / (k - 1)))

    out: dict = {}
    out["Overall"] = round(
        _one(rna.obsm[embed_key], atac.obsm[embed_key]), ndigits,
    )
    for tp in _resolve_samples(rna, samples, sample_key):
        sub_r = rna[rna.obs[sample_key] == tp].obsm[embed_key]
        sub_a = atac[atac.obs[sample_key] == tp].obsm[embed_key]
        out[tp] = round(_one(sub_r, sub_a), ndigits)
    return out


def consistency_ari(
        rna: AnnData, atac: AnnData,
        domain_key: str = "domain",
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""ARI between RNA-derived and ATAC-derived cluster labels per sample.

    Expects ``rna.obs[domain_key]`` and ``atac.obs[domain_key]`` to be the
    modality-specific labels (e.g. from
    :func:`storm.clustering.concat_clustering`). High ARI means the two
    modalities partition the tissue into the same spatial domains.
    """
    out: dict = {}
    scores: List[float] = []
    for tp in _resolve_samples(rna, samples, sample_key):
        r = rna[rna.obs[sample_key] == tp].obs[domain_key]
        a = atac[atac.obs[sample_key] == tp].obs[domain_key]
        ari = adjusted_rand_score(r, a)
        scores.append(ari)
        out[tp] = round(ari, ndigits)
    out["Overall"] = round(float(np.mean(scores)), ndigits) if scores else float("nan")
    return out


# ---------------------------------------------------------------------------
# Batch / timepoint integration
# ---------------------------------------------------------------------------

def pcr_score(
        adata: AnnData,
        embed_key: str = "X_storm",
        batch_key: str = "timepoint",
        n_comps: int = 50,
        ndigits: int = 4,
) -> dict:
    r"""Principal-component regression batch-integration score.

    Returns ``1 - scib.metrics.pcr(...)`` so higher = better integration.
    Requires `scib`.
    """
    try:
        import scib
    except ImportError as exc:
        raise ImportError(
            "storm.evaluation.pcr_score requires `scib` "
            "(https://github.com/theislab/scib)."
        ) from exc
    score = scib.metrics.pcr(
        adata,
        covariate=batch_key,
        embed=embed_key,
        n_comps=n_comps,
        recompute_pca=False,
        verbose=False,
    )
    return {"Overall": round(1 - score, ndigits)}


def blisi(
        adata: AnnData,
        embed_key: str = "X_storm",
        batch_key: str = "timepoint",
        k: int = 15,
        ndigits: int = 4,
) -> dict:
    r"""Batch-LISI on the joint embedding (higher = better batch mixing)."""
    embedding = adata.obsm[embed_key]
    batches = adata.obs[batch_key].astype(str).values
    distances = pairwise_distances(embedding, metric="euclidean")
    scores = []
    for i in range(distances.shape[0]):
        nn = np.argsort(distances[i])[:k]
        _, counts = np.unique(batches[nn], return_counts=True)
        scores.append(1.0 / np.sum((counts / k) ** 2))
    rescaled = (np.asarray(scores) - 1) / (k - 1)
    return {"Overall": round(float(np.mean(rescaled)), ndigits)}


# ---------------------------------------------------------------------------
# Cell-type recovery
# ---------------------------------------------------------------------------

def nmi_score(
        adata: AnnData,
        domain_key: str,
        ref_key: str,
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        implementation: str = "arithmetic",
        ndigits: int = 4,
) -> dict:
    r"""scib-style NMI between cluster ``domain_key`` and ground-truth ``ref_key``.

    Computed per sample and overall. Requires `scib`.
    """
    try:
        from scib.metrics import nmi
    except ImportError as exc:
        raise ImportError(
            "storm.evaluation.nmi_score requires `scib` "
            "(https://github.com/theislab/scib)."
        ) from exc
    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        sub = adata[adata.obs[sample_key] == tp]
        out[tp] = round(
            nmi(sub, domain_key, ref_key, implementation=implementation),
            ndigits,
        )
    out["Overall"] = round(
        nmi(adata, domain_key, ref_key, implementation=implementation),
        ndigits,
    )
    return out


# ---------------------------------------------------------------------------
# Niche coherence
# ---------------------------------------------------------------------------

def lisi(
        adata: AnnData,
        embed_key: str = "X_storm",
        annotation_key: str = "RNA_clusters",
        k: int = 15,
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""Cell-type Local Inverse Simpson Index for niche coherence.

    Returns ``(k - LISI) / (k - 1)`` so higher = sharper local
    cell-type concentration.
    """
    def _one(sub: AnnData) -> float:
        embedding = sub.obsm[embed_key]
        annotations = sub.obs[annotation_key].astype(str).values
        distances = pairwise_distances(embedding, metric="euclidean")
        scores = []
        for i in range(distances.shape[0]):
            nn = np.argsort(distances[i])[:k]
            _, counts = np.unique(annotations[nn], return_counts=True)
            scores.append(1.0 / np.sum((counts / k) ** 2))
        return float(np.mean((k - np.asarray(scores)) / (k - 1)))

    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        out[tp] = round(_one(adata[adata.obs[sample_key] == tp]), ndigits)
    out["Overall"] = round(_one(adata), ndigits)
    return out


def cell_type_asw(
        adata: AnnData,
        embed_key: str = "X_storm",
        label_key: str = "RNA_clusters",
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""Average silhouette width by cell type, per sample + overall ([0, 1])."""
    def _asw(X: np.ndarray, labels) -> float:
        if len(np.unique(labels)) < 2:
            return float("nan")
        sil = silhouette_samples(X, labels)
        return float(0.5 * (np.mean(sil) + 1))

    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        sub = adata[adata.obs[sample_key] == tp]
        out[tp] = round(
            _asw(sub.obsm[embed_key], sub.obs[label_key].values),
            ndigits,
        )
    out["Overall"] = round(
        _asw(adata.obsm[embed_key], adata.obs[label_key].values),
        ndigits,
    )
    return out


def map_score(
        adata: AnnData,
        embed_key: str = "X_storm",
        label_key: str = "RNA_clusters",
        k_frac: float = 0.01,
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""Mean average precision of label retrieval among k-NN."""
    def _one(sub: AnnData) -> float:
        X = sub.obsm[embed_key]
        labels = sub.obs[label_key].values
        k = max(1, int(k_frac * len(labels)))
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        knn = nbrs.kneighbors(X, return_distance=False)[:, 1:]
        ap = []
        for i in range(len(labels)):
            mask = labels[knn[i]] == labels[i]
            if mask.sum() == 0:
                ap.append(0.0)
            else:
                precisions = [
                    mask[: j + 1].sum() / (j + 1)
                    for j in range(k)
                    if mask[j]
                ]
                ap.append(float(np.mean(precisions)) if precisions else 0.0)
        return float(np.mean(ap))

    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        out[tp] = round(_one(adata[adata.obs[sample_key] == tp]), ndigits)
    out["Overall"] = round(_one(adata), ndigits)
    return out


# ---------------------------------------------------------------------------
# Spatial conservation
# ---------------------------------------------------------------------------

def global_morans_i(
        adata: AnnData,
        embed_key: str = "X_storm",
        spatial_key: str = "spatial",
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        n_perms: int = 100,
        ndigits: int = 4,
) -> dict:
    r"""Global Moran's I averaged across embedding dimensions, per sample + overall.

    Requires `squidpy`.
    """
    try:
        import squidpy as sq
    except ImportError as exc:
        raise ImportError(
            "storm.evaluation.global_morans_i requires `squidpy` "
            "(https://github.com/scverse/squidpy)."
        ) from exc

    num_dims = adata.obsm[embed_key].shape[1]
    emb_names = [f"{embed_key}_{i}" for i in range(num_dims)]

    def _one(latent: np.ndarray, spatial: np.ndarray) -> float:
        tmp = ad.AnnData(X=latent)
        tmp.var_names = emb_names
        tmp.obsm[spatial_key] = spatial
        sq.gr.spatial_neighbors(tmp, coord_type="generic", spatial_key=spatial_key)
        sq.gr.spatial_autocorr(tmp, mode="moran", genes=emb_names, n_perms=n_perms)
        return float(tmp.uns["moranI"]["I"].mean())

    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        sub = adata[adata.obs[sample_key] == tp]
        out[tp] = round(
            _one(sub.obsm[embed_key], sub.obsm[spatial_key]), ndigits,
        )
    out["Overall"] = round(
        _one(adata.obsm[embed_key], adata.obsm[spatial_key]), ndigits,
    )
    return out


def clisis(
        adata: AnnData,
        embed_key: str = "X_storm",
        spatial_key: str = "spatial",
        cell_type_key: str = "RNA_clusters",
        k: int = 15,
        samples: Optional[Iterable[str]] = None,
        sample_key: str = "timepoint",
        ndigits: int = 4,
) -> dict:
    r"""Cell-type LISI similarity (CLISIS) between latent and spatial spaces.

    Compares the local cell-type heterogeneity in the latent embedding to
    that in the spatial coordinates; returns
    ``1 - median(|log(LISI_latent / LISI_spatial)|)`` rescaled to [0, 1].
    """
    def _lisi(emb: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(emb, metric="euclidean")
        scores = []
        for i in range(distances.shape[0]):
            nn = np.argsort(distances[i])[:k]
            _, counts = np.unique(cell_types[nn], return_counts=True)
            scores.append(1.0 / np.sum((counts / k) ** 2))
        return np.asarray(scores)

    def _one(sub: AnnData) -> float:
        latent_lisi = _lisi(sub.obsm[embed_key], sub.obs[cell_type_key].values)
        spatial_lisi = _lisi(sub.obsm[spatial_key], sub.obs[cell_type_key].values)
        rel = np.log(latent_lisi / spatial_lisi)
        norm = rel / np.max(np.abs(rel))
        return float(1 - np.median(np.abs(norm)))

    out: dict = {}
    for tp in _resolve_samples(adata, samples, sample_key):
        out[tp] = round(_one(adata[adata.obs[sample_key] == tp]), ndigits)
    return out


# ---------------------------------------------------------------------------
# High-level harness
# ---------------------------------------------------------------------------

# Metric registry: name → (callable, family, needs_extra_dep). Used by
# :func:`benchmark` so a single ``include`` / ``exclude`` filter can pick a
# subset.
_METRIC_REGISTRY: Tuple[Tuple[str, str], ...] = (
    ("FOSCTTM",                "integration"),
    ("MLISI",                  "integration"),
    ("Consistency",            "integration"),
    ("Joint PCR",              "batch"),
    ("Joint BLISI",            "batch"),
    ("Joint NMI",              "recovery"),
    ("Joint LISI",             "niche"),
    ("Joint Cell Type ASW",    "niche"),
    ("Joint MAP",              "niche"),
    ("Joint Global Morans I",  "spatial"),
    ("Joint CLISIS",           "spatial"),
)


def benchmark(
        rna: AnnData,
        atac: AnnData,
        joint: Optional[AnnData] = None,
        *,
        embed_key: str = "X_storm",
        domain_key: str = "domain",
        sample_key: str = "timepoint",
        samples: Optional[Iterable[str]] = None,
        ref_keys: Sequence[str] = ("RNA_clusters", "ATAC_clusters"),
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        n_clusters: Optional[int] = None,
) -> pd.DataFrame:
    r"""Run the full STORM benchmark battery and return a tidy DataFrame.

    Mirrors ``bench_range.calc_bench`` but with a flat return shape:
    one row per (metric, sample), columns ``["Metric", "n_clusters",
    "Sample", "Value"]``.

    Parameters
    ----------
    rna, atac
        Per-modality, paired AnnDatas with the embedding at
        ``adata.obsm[embed_key]`` and modality-specific cluster labels at
        ``adata.obs[domain_key]`` (e.g. from
        :func:`storm.clustering.concat_clustering`).
    joint
        Joint AnnData with one row per spatial location, e.g. from
        :func:`storm.clustering.joint_clustering`. Required for the
        ``"batch"``, ``"recovery"``, ``"niche"``, and ``"spatial"``
        metric families. Pass ``None`` to skip them.
    embed_key, domain_key, sample_key
        AnnData keys to read.
    samples
        Iterable of sample IDs to evaluate (defaults to
        ``adata.obs[sample_key].unique()``).
    ref_keys
        Ground-truth cell-type label columns in ``joint.obs`` used by
        the NMI / LISI / ASW / MAP / CLISIS metrics.
    include, exclude
        Filter metric names (case-insensitive substring match against
        :data:`_METRIC_REGISTRY`). ``include`` takes precedence.
    n_clusters
        Optional integer, written into the returned DataFrame's
        ``n_clusters`` column. Defaults to the unique cluster count in
        ``rna.obs[domain_key]`` if available, else ``None``.

    Returns
    -------
    df
        Long-form ``DataFrame``. Use ``df.pivot_table(...)`` if you want
        the bench_range-style nested layout.
    """
    if n_clusters is None and domain_key in rna.obs.columns:
        try:
            n_clusters = int(rna.obs[domain_key].nunique())
        except Exception:
            n_clusters = None

    def _wanted(name: str) -> bool:
        if include is not None:
            return any(s.lower() in name.lower() for s in include)
        if exclude is not None:
            return not any(s.lower() in name.lower() for s in exclude)
        return True

    samples_list = _resolve_samples(rna, samples, sample_key)
    rows: List[dict] = []

    def _push(metric: str, scores: Mapping[str, float]) -> None:
        for sample, value in scores.items():
            rows.append({
                "Metric": metric,
                "n_clusters": n_clusters,
                "Sample": sample,
                "Value": value,
            })

    # ---- Multi-omics integration ---------------------------------------
    if _wanted("FOSCTTM"):
        _push("FOSCTTM", foscttm_paired(
            rna, atac, embed_key=embed_key,
            samples=samples_list, sample_key=sample_key,
        ))
    if _wanted("MLISI"):
        _push("MLISI", mlisi(
            rna, atac, embed_key=embed_key,
            samples=samples_list, sample_key=sample_key,
        ))
    if _wanted("Consistency") and domain_key in rna.obs.columns:
        _push("Consistency", consistency_ari(
            rna, atac, domain_key=domain_key,
            samples=samples_list, sample_key=sample_key,
        ))

    # ---- Joint AnnData metrics -----------------------------------------
    if joint is None:
        return pd.DataFrame(rows)

    if _wanted("Joint PCR"):
        _push("Joint PCR", pcr_score(
            joint, embed_key=embed_key, batch_key=sample_key,
        ))
    if _wanted("Joint BLISI"):
        _push("Joint BLISI", blisi(
            joint, embed_key=embed_key, batch_key=sample_key,
        ))

    for ref in ref_keys:
        if ref not in joint.obs.columns:
            continue
        if _wanted("Joint NMI"):
            _push(f"Joint NMI with {ref}", nmi_score(
                joint, domain_key=domain_key, ref_key=ref,
                samples=samples_list, sample_key=sample_key,
            ))
        if _wanted("Joint LISI"):
            _push(f"Joint LISI with {ref}", lisi(
                joint, embed_key=embed_key, annotation_key=ref,
                samples=samples_list, sample_key=sample_key,
            ))
        if _wanted("Joint Cell Type ASW"):
            _push(f"Joint Cell Type ASW with {ref}", cell_type_asw(
                joint, embed_key=embed_key, label_key=ref,
                samples=samples_list, sample_key=sample_key,
            ))
        if _wanted("Joint MAP"):
            _push(f"Joint MAP with {ref}", map_score(
                joint, embed_key=embed_key, label_key=ref,
                samples=samples_list, sample_key=sample_key,
            ))
        if _wanted("Joint CLISIS"):
            _push(f"Joint CLISIS with {ref}", clisis(
                joint, embed_key=embed_key, cell_type_key=ref,
                samples=samples_list, sample_key=sample_key,
            ))

    if _wanted("Joint Global Morans I"):
        _push("Joint Global Moran's I", global_morans_i(
            joint, embed_key=embed_key,
            samples=samples_list, sample_key=sample_key,
        ))

    return pd.DataFrame(rows)
