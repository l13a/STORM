r"""
STORM integration models.

The user-facing entry points are :class:`STORMModel` and
:class:`PairedSTORMModel` (for spatially paired multi-omics), plus the
high-level :func:`fit_STORM` helper that runs the pretrain → balance →
fine-tune loop.
"""

import os
from pathlib import Path
from typing import Mapping, Optional

import dill
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from ..data import estimate_balancing_weight
from ..typehint import Kws
from .._internal import config, logged
from .base import Model
from .dx import integration_consistency
from .nn import autodevice
from .storm import STORM_DEFAULT_KEYS, PairedSTORMModel, STORMModel
from typing import Tuple


@logged
def configure_dataset(
        adata: AnnData, prob_model: str,
        use_highly_variable: bool = True,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_adj: str = "spatial_adj",
        use_batch: Optional[str] = None,
        use_cell_type: Optional[str] = None,
        use_dsc_weight: Optional[str] = None,
        use_timepoint: Optional[str] = None,
        use_obs_names: bool = False
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    prob_model
        Probabilistic generative model used by the decoder,
        must be one of ``{"Normal", "ZIN", "ZILN", "NB", "ZINB"}``.
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_adj
        Adjacency matrix from the spatial coordinates (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_timepoint
        Data timpoint information (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if config.ANNDATA_KEY in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    data_config["prob_model"] = prob_model
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = pd.Index(
            adata.obs[use_cell_type]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None
    if use_dsc_weight:
        if use_dsc_weight not in adata.obs:
            raise ValueError("Invalid `use_dsc_weight`!")
        data_config["use_dsc_weight"] = use_dsc_weight
    else:
        data_config["use_dsc_weight"] = None
    if use_timepoint:
        if use_timepoint not in adata.obs:
            raise ValueError("Invalid `use_timepoint`!")
        data_config["use_timepoint"] = use_timepoint
        data_config["timepoints"] = pd.Index(
            adata.obs[use_timepoint]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_timepoint"] = None
        data_config["timepoints"] = None

    data_config["use_obs_names"] = use_obs_names
    data_config["adjacency_key"] = use_adj
    adata.uns[config.ANNDATA_KEY] = data_config


def load_model(fname: os.PathLike) -> Model:
    r"""
    Load model from file

    Parameters
    ----------
    fname
        Specifies path to the file

    Returns
    -------
    model
        Loaded model
    """
    fname = Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    model.upgrade()  # pylint: disable=no-member
    model.net.device = autodevice()  # pylint: disable=no-member
    return model


@logged
def fit_STORM(
        adatas: Mapping[str, AnnData], graph: nx.Graph, model: type = STORMModel,
        init_kws: Kws = None, compile_kws: Kws = None, fit_kws: Kws = None,
        balance_kws: Kws = None, finetune_compile_kws: Kws = None,
) -> STORMModel:
    r"""
    Fit GLUE model to integrate single-cell multi-omics data

    Parameters
    ----------
    adatas
        Single-cell datasets (indexed by modality name)
    graph
        Guidance graph
    model
        Model class, must be one of
        {:class:`storm.models.storm.STORMModel`,
        :class:`storm.models.storm.PairedSTORMModel`}
    init_kws
        Model initialization keyword arguments
        (see the constructor of the ``model`` class,
        either :class:`storm.models.storm.STORMModel`
        or :class:`storm.models.storm.PairedSTORMModel`)
    compile_kws
        Model compile keyword arguments
        (see the ``compile`` method of the ``model`` class,
        either :meth:`storm.models.storm.STORMModel.compile`
        or :meth:`storm.models.storm.PairedSTORMModel.compile`)
    fit_kws
        Model fitting keyword arguments
        (see :meth:`storm.models.storm.STORMModel.fit`)
    balance_kws
        Balancing weight estimation keyword arguments
        (see :func:`storm.data.estimate_balancing_weight`)
    finetune_compile_kws
        Compile keyword arguments applied on top of ``compile_kws`` for the
        fine-tuning stage. Defaults to
        ``{"n_epochs_rna": 1, "n_epochs_atac": 1, "lam_tmp": 100.,
        "lam_align": 1000.}``; pass a dict to override any of these (or add
        further compile kwargs) for fine-tuning without affecting pretraining.

    Returns
    -------
    model
        Fitted model object
    """
    init_kws = init_kws or {}
    compile_kws = compile_kws or {}
    fit_kws = fit_kws or {}
    balance_kws = balance_kws or {}
    finetune_compile_defaults = {
        "n_epochs_rna": 1,
        "n_epochs_atac": 1,
        "lam_tmp": 100.,
        "lam_align": 1000.,
    }
    finetune_compile_defaults.update(finetune_compile_kws or {})
    finetune_compile_kws = finetune_compile_defaults

    fit_STORM.logger.info("Pretraining STORM model...")
    pretrain_init_kws = init_kws.copy()
    pretrain_init_kws.update({"shared_batches": False})
    pretrain_fit_kws = fit_kws.copy()
    # pretrain_fit_kws.update({"align_burnin": np.inf, "safe_burnin": False})
    if "directory" in pretrain_fit_kws:
        pretrain_fit_kws["directory"] = \
            os.path.join(pretrain_fit_kws["directory"], "pretrain")
    pretrain = model(adatas, sorted(graph.nodes), **pretrain_init_kws)
    pretrain.compile(**compile_kws)
    pretrain.fit(adatas, graph, **pretrain_fit_kws)
    if "directory" in pretrain_fit_kws:
        pretrain.save(os.path.join(pretrain_fit_kws["directory"], "pretrain.dill"))

    fit_STORM.logger.info("Estimating balancing weight...")
    for k, adata in adatas.items():
        adata.obsm[f"X_{config.TMP_PREFIX}"] = pretrain.encode_data(k, adata)
    if init_kws.get("shared_batches"):
        use_batch = set(
            adata.uns[config.ANNDATA_KEY]["use_batch"]
            for adata in adatas.values()
        )
        use_batch = use_batch.pop() if len(use_batch) == 1 else None
    else:
        use_batch = None
    estimate_balancing_weight(
        *adatas.values(), use_rep=f"X_{config.TMP_PREFIX}", use_batch=use_batch,
        key_added="balancing_weight", **balance_kws
    )
    for adata in adatas.values():
        adata.uns[config.ANNDATA_KEY]["use_dsc_weight"] = "balancing_weight"
        del adata.obsm[f"X_{config.TMP_PREFIX}"]

    fit_STORM.logger.info("Fine-tuning STORM model...")
    finetune_fit_kws = fit_kws.copy()
    if "directory" in finetune_fit_kws:
        finetune_fit_kws["directory"] = \
            os.path.join(finetune_fit_kws["directory"], "fine-tune")

    finetune = model(adatas, sorted(graph.nodes), **init_kws)
    finetune.adopt_pretrained_model(pretrain)

    # Apply fine-tuning compile overrides on top of the pretrain compile kwargs.
    # n_epochs_atac is ignored for single-modality runs.
    finetune_compile = {**compile_kws, **finetune_compile_kws}

    finetune.compile(**finetune_compile)
    fit_STORM.logger.debug("Increasing random seed by 1 to prevent idential data order...")
    finetune.random_seed += 1
    finetune.fit(adatas, graph, **finetune_fit_kws)
    if "directory" in finetune_fit_kws:
        finetune.save(os.path.join(finetune_fit_kws["directory"], "fine-tune.dill"))

    return finetune
