r"""
STORM core model module.

Defines the user-facing :class:`STORMModel` and :class:`PairedSTORMModel`
classes used for graph-linked unified embedding of spatial multi-omics
data.
"""

import copy
import os
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple, Union, Literal

import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
import torch.nn as nn

from ..graph import check_graph
from ..num import normalize_edges
from .._internal import AUTO, config, get_chained_attr, logged
from . import sc
from .base import Model
from .dataset import AnnDataset, ArrayDataset, DataLoader, GraphDataset
from .glue import GLUE, GLUETrainer
from .nn import freeze_running_stats

from .losses import (compute_omics_recon_nb_loss, 
                    compute_gp_l1_reg_loss,
                    negative_sampling_loss)

from sklearn.decomposition import PCA
import numpy as np
import rpy2
import ot
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


#---------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)


#----------------------------- Network definition ------------------------------

class STORM(GLUE):

    r"""
    GLUE network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x_targets
        Data decoders (indexed by modality name) for targets
    u2x_sources
        Data decoders (indexed by modality name) for sources
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality and Timepoint combined discriminator
    node_label_aggregator
        Aggregator for neighbour nodes
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2z: Mapping[str, sc.SimpleDataEncoder],
            u2x_targets: Mapping[str, sc.MaskedOmicsFeatureDecoder],
            u2x_sources: Mapping[str, sc.MaskedOmicsFeatureDecoder],
            graph_decoder: sc.CosineSimGraphDecoder,
            idx: Mapping[str, torch.Tensor],
            du: sc.CombinedDiscriminator, prior: sc.Prior,
            node_label_aggregator: Mapping[str, sc.OneHopGCNNormNodeLabelAggregator],
            u2c: Optional[sc.Classifier],
            n_input: int,
            n_hidden_encoder: int,
            n_prior_gp: int,
            n_addon_gp: int,
            n_output_genes: int,
            n_output_peaks: int,
            target_rna_decoder_mask: torch.Tensor,
            source_rna_decoder_mask: torch.Tensor,
            target_atac_decoder_mask: Optional[torch.Tensor],
            source_atac_decoder_mask: Optional[torch.Tensor],
            target_rna_decoder_addon_mask: torch.Tensor,
            source_rna_decoder_addon_mask: torch.Tensor,
            target_atac_decoder_addon_mask: Optional[torch.Tensor],
            source_atac_decoder_addon_mask: Optional[torch.Tensor],
            features_idx_dict: dict,
            features_scale_factors: torch.Tensor,
            gene_peaks_mask: Optional[torch.Tensor],
            active_gp_thresh_ratio: float,
            active_gp_type: Literal["mixed", "separate"],
            include_edge_recon_loss: bool,
            include_edge_kl_loss: bool
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2z, u2x_targets, u2x_sources, graph_decoder, idx, du, prior, node_label_aggregator,
        n_input=n_input,
        n_hidden_encoder=n_hidden_encoder,
        n_prior_gp=n_prior_gp,
        n_addon_gp=n_addon_gp,
        n_output_genes=n_output_genes,
        n_output_peaks=n_output_peaks,
        target_rna_decoder_mask=target_rna_decoder_mask,
        source_rna_decoder_mask=source_rna_decoder_mask,
        target_atac_decoder_mask=target_atac_decoder_mask,
        source_atac_decoder_mask=source_atac_decoder_mask,
        target_rna_decoder_addon_mask=target_rna_decoder_addon_mask,
        source_rna_decoder_addon_mask=source_rna_decoder_addon_mask,
        target_atac_decoder_addon_mask=target_atac_decoder_addon_mask,
        source_atac_decoder_addon_mask=source_atac_decoder_addon_mask,
        features_idx_dict=features_idx_dict,
        features_scale_factors=features_scale_factors,
        gene_peaks_mask=gene_peaks_mask,
        active_gp_thresh_ratio=active_gp_thresh_ratio,
        active_gp_type=active_gp_type,
        include_edge_recon_loss=include_edge_recon_loss,
        include_edge_kl_loss=include_edge_kl_loss)
        self.u2c = u2c.to(self.device) if u2c else None
    
    @torch.no_grad()
    def get_gp_weights(self,
                       only_masked_features: bool=False,
                       gp_type: Literal["all", "prior", "addon"]="all"
                       ) -> List[torch.Tensor]:
        """
        Get the gene program weights of the omics feature decoders.

        Returns:
        ----------
        gp_weights_all_modalities:
            List of tensors containing the decoder gp weights for each
            omics modality (dim: (n_prior_gp + n_addon_gp) x n_omics_features)
        """
        gp_weights_all_modalities = []

        for modality in self.keys:
            target_decoder = self.u2x_targets[f"{modality}"]
            source_decoder = self.u2x_sources[f"{modality}"]

            if gp_type != "addon":
                # Get decoder weights of masked gps
                target_gp_weights_masked = (
                    target_decoder.nb_means_normalized_decoder.masked_l.weights[0].data
                    ).clone()
                source_gp_weights_masked = (
                    source_decoder.nb_means_normalized_decoder.masked_l.weights[0].data
                    ).clone()
                gp_weights = torch.cat((target_gp_weights_masked,
                                        source_gp_weights_masked),
                                    dim=0)

            # Add decoder weights of addon gps
            if (gp_type != "masked") & (self.n_addon_gp_ > 0):
                target_gp_weights_addon = (
                    target_decoder.nb_means_normalized_decoder.addon_l.weights[0].data
                    ).clone()
                source_gp_weights_addon = (
                    source_decoder.nb_means_normalized_decoder.addon_l.weights[0].data
                    ).clone()
                gp_weights_addon = torch.cat((target_gp_weights_addon,
                                              source_gp_weights_addon),
                                             dim=0)
                
            if (gp_type == "all") & (self.n_addon_gp_ > 0):
                gp_weights = torch.cat([gp_weights, gp_weights_addon], axis=1)
            elif gp_type == "addon":
                gp_weights = gp_weights_addon

            # Only keep omics features in mask
            if only_masked_features:
                mask_idx = getattr(self, "features_idx_dict_")[
                    f"masked_{modality}_idx"]
                gp_weights = gp_weights[mask_idx, :]
            
            # Append current modality to output list
            gp_weights_all_modalities.append(gp_weights)
        return gp_weights_all_modalities

    @torch.no_grad()
    def get_active_gp_mask(
            self,
            abs_gp_weights_agg_mode: Literal["sum",
                                             "nzmeans",
                                             "sum+nzmeans",
                                             "nzmedians",
                                             "sum+nzmedians"]="sum+nzmeans",
            return_gp_weights: bool=False,
            normalize_gp_weights_with_features_scale_factors: bool=False,
            which_modality: str = "rna"
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a mask of active gene programs based on the rna decoder gene weights
        of gene programs. Active gene programs are gene programs whose absolute
        gene weights aggregated over all genes are greater than
        ´self.active_gp_thresh_ratio_´ times the absolute gene weights
        aggregation of the gene program with the maximum value across all gene
        programs. Depending on ´abs_gp_weights_agg_mode´, the aggregation will
        be either a sum of absolute gene weights (prioritizes gene programs that
        reconstruct many genes) or a mean of non-zero absolute gene weights
        (normalizes for the number of genes that a gene program reconstructs) or
        a combination of the two.

        Parameters
        ----------
        abs_gp_weights_agg_mode:
            If ´sum´, uses sums of absolute gp weights for aggregation and
            active gp determination. If ´nzmeans´, uses means of non-zero
            absolute gp weights for aggregation and active gp determination. If
            ´sum+nzmeans´, uses a combination of sums and means of non-zero
            absolute gp weights for aggregation and active gp determination.
        return_gp_weights:
            If ´True´, in addition return the rna decoder gene weights of the
            active gene programs.

        Returns
        ----------
        active_gp_mask:
            Boolean tensor of gene programs which contains `True` for active
            gene programs and `False` for inactive gene programs.
        active_gp_weights:
            Tensor containing the rna decoder gene weights of active gene
            programs.
        """
        # TODO: handle different devices
        # device = next(self.parameters()).device
        device = torch.device('cuda')
        
        active_gp_mask = torch.zeros(self.n_prior_gp_ + self.n_addon_gp_,
                                     dtype=torch.bool,
                                     device=device)

        if self.active_gp_type_ == "mixed":
            gp_types = ["all"]
        elif (self.n_addon_gp_ > 0):
            gp_types = ["masked", "addon"]
        else:
            gp_types = ["masked"]

        for gp_type in gp_types:
            if which_modality == "rna":
                gp_weights = self.get_gp_weights(only_masked_features=False,
                                             gp_type=gp_type)[0]
            elif which_modality == "atac":
                gp_weights = self.get_gp_weights(only_masked_features=False,
                                             gp_type=gp_type)[1]
            # Get index of gps based on ´gp_type´
            if gp_type == "masked":
                gp_idx = slice(None, self.n_prior_gp_)
            elif gp_type == "addon":
                gp_idx = slice(self.n_prior_gp_, None)
            elif gp_type == "all":
                gp_idx = slice(None, None)
            
            # Normalize gp weights with features scale factors
            # TODO: currently never normalizing --> figure out when need normalize
            if normalize_gp_weights_with_features_scale_factors:
                gp_weights_normalized = (gp_weights /
                                         self.features_scale_factors_[:, None].to(device))
            else:
                gp_weights_normalized = gp_weights
            
            if which_modality == "rna":
                # Normalize gp weights with running mean absolute gp scores
                gp_weights_normalized = (self.running_mean_abs_mu[gp_idx] *
                                        gp_weights_normalized)
            elif which_modality == "atac":
                gp_weights_normalized = (self.running_mean_abs_mu_atac[gp_idx] *
                                        gp_weights_normalized)

            # Aggregate absolute normalized gp weights based on
            # ´abs_gp_weights_agg_mode´ and calculate thresholds of aggregated
            # absolute normalized gp weights and get active gp mask and (optionally)
            # active gp weights
            abs_gp_weights_sums = gp_weights_normalized.norm(p=1, dim=0)
            if abs_gp_weights_agg_mode in ["sum", "sum+nzmeans", "sum+nzmedians"]:
                max_abs_gp_weights_sum = max(abs_gp_weights_sums)
                min_abs_gp_weights_sum_thresh = (self.active_gp_thresh_ratio_ * 
                                                max_abs_gp_weights_sum)
                active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                    abs_gp_weights_sums >= min_abs_gp_weights_sum_thresh)
            
            if abs_gp_weights_agg_mode in ["nzmeans", "sum+nzmeans"]:
                abs_gp_weights_nzmeans = (
                    abs_gp_weights_sums / 
                    torch.count_nonzero(gp_weights_normalized, dim=0))
                abs_gp_weights_nzmeans = torch.nan_to_num(abs_gp_weights_nzmeans)
                max_abs_gp_weights_nzmean = max(abs_gp_weights_nzmeans)
                min_abs_gp_weights_nzmean_thresh = (self.active_gp_thresh_ratio_ *
                                                    max_abs_gp_weights_nzmean)
                if abs_gp_weights_agg_mode == "nzmeans":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >= 
                        min_abs_gp_weights_nzmean_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmeans":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >=
                        min_abs_gp_weights_nzmean_thresh)
            if abs_gp_weights_agg_mode in ["nzmedians", "sum+nzmedians"]:
                zero_mask = (gp_weights_normalized == 0)
                abs_gp_weights_normalized_with_nan = torch.where(zero_mask, torch.tensor(float('nan')), torch.abs(gp_weights_normalized))
                abs_gp_weights_nzmedians = torch.nanmedian(abs_gp_weights_normalized_with_nan, dim=0).values
                abs_gp_weights_nzmedians = torch.nan_to_num(abs_gp_weights_nzmedians)
                max_abs_gp_weights_nzmedian = torch.max(abs_gp_weights_nzmedians)
                min_abs_gp_weights_nzmedian_thresh = (0.01 *
                                                      max_abs_gp_weights_nzmedian)
                if abs_gp_weights_agg_mode == "nzmedians":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >= 
                        min_abs_gp_weights_nzmedian_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmedians":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >=
                        min_abs_gp_weights_nzmedian_thresh)
        if return_gp_weights:
            active_gp_weights = gp_weights[:, active_gp_mask]
            return active_gp_mask, active_gp_weights
        else:
            return active_gp_mask
    

#----------------------------- Trainer definition ------------------------------

# TODO: fix up this to match PairedDataTensors
DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to STORMTrainer.compute_losses


@logged
class STORMTrainer(GLUETrainer):

    r"""
    Trainer for :class:`STORM`

    Parameters
    ----------
    net
        :class:`STORM` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration

    def __init__(
            self, net: STORM, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, 
            lam_cos: float = None, lam_masked_l1: float = None, lam_addon_l1: float = None,
            lam_adj: float = None,
            lam_tmp: float = None,
            n_epochs_rna: int = None,
            n_epochs_atac: int = None,
            n_epochs_all_gp: int = None,
            normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl, 
            lam_graph=lam_graph, lam_align=lam_align, 
            lam_cos=lam_cos, lam_masked_l1=lam_masked_l1, lam_addon_l1=lam_addon_l1,
            lam_adj=lam_adj,
            lam_tmp=lam_tmp,
            n_epochs_rna = n_epochs_rna,
            n_epochs_atac = n_epochs_atac,
            n_epochs_all_gp = n_epochs_all_gp,
            modality_weight=modality_weight,
            optim=optim, lr=lr, **kwargs
        )
        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.normalize_u = normalize_u
        self.freeze_u = False
        if net.u2c:
            self.required_losses.append("sup_loss")
            self.vae_optim = getattr(torch.optim, optim)(
                chain(
                    self.net.g2v.parameters(),
                    self.net.v2g.parameters(),
                    self.net.x2u.parameters(),
                    self.net.u2x_targets.parameters(),
                    self.net.u2x_sources.parameters(),
                    self.net.u2c.parameters()
                ), lr=self.lr, **kwargs
            )
        if "atac" in net.keys:
            self.lam_cos = lam_cos
            self.required_losses += ["cos_loss"]
        else:
            self.lam_cos = None

        self.required_losses.append("adj_loss")
        self.required_losses.append("masked_gp_l1_loss")
        if self.net.n_addon_gp_ > 0:
            self.required_losses.append("addon_gp_l1_loss")
        self.required_losses.append("dsc_loss_t")


    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], \
            data[5*K+1:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False, 
            rna_only: bool = False, atac_only: bool = False,
            use_only_active_gps: bool = False, return_agg_weights: bool=False,
            update_atac_dynamic_decoder_mask: bool=False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        
        # Normal(loc: 7299 x 50, scale: 7299 x 50)
        v = net.g2v(self.eidx, self.enorm, self.esgn)
        # 7299 x 50
        vsamp = v.rsample()

        zsamp = {}
        for k in net.keys:
            # 128 x 178
            zsamp[k] = net.u2z[k](usamp[k])

        if use_only_active_gps:
            active_gp_mask = net.get_active_gp_mask()
            # Set gp scores of inactive gene programs to 0 to not affect 
            # graph decoder
            for k in net.keys:
                zsamp[k] = zsamp[k].clone() 
                zsamp[k][:, ~active_gp_mask] = 0 

        with torch.no_grad():
            if net.training:
                # Update running mean absolute gp scores using exponential
                # moving average with momentum of 0.1
                # 50
                mean_abs_mu = u["rna"].mean.norm(p=1, dim=0) / u["rna"].mean.size(0)
                # 178
                mean_abs_mu = net.u2z["rna"](mean_abs_mu)
                # running_mean_abs_mu = N_gp = 178
                net.running_mean_abs_mu = (
                    0.1 * mean_abs_mu + 0.9 * net.running_mean_abs_mu.to(net.device))
            if use_only_active_gps:
                # Set running mean abs mu of inactive gene programs to 0 for
                # active gp determination
                net.running_mean_abs_mu[~active_gp_mask] = 0  

                # Set dynamic mask to 0 for all inactive gene programs to
                # not affect omics decoders
                net.target_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0
                net.source_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0

                if len(net.keys) == 2:
                    net.target_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    net.source_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0

        # Determine which features should be reconstructed based on
        # static and dynamic masks (if a feature is not connected to any
        # node it should not be reconstructed to not influence softmax
        # activation outputs). This can happen when no add-on gene programs
        # are present or when gene programs are turned off.
        if net.n_addon_gp_ > 0:
            target_rna_decoder_static_mask = torch.cat(
                (net.target_rna_decoder_mask,
                    net.target_rna_decoder_addon_mask[0]), dim=0)
            source_rna_decoder_static_mask = torch.cat(
                (net.source_rna_decoder_mask,
                    net.source_rna_decoder_addon_mask[0]), dim=0)
        else:
            target_rna_decoder_static_mask = net.target_rna_decoder_mask
            source_rna_decoder_static_mask = net.source_rna_decoder_mask

        self.target_n_gps_per_gene = (
            target_rna_decoder_static_mask
            * net.target_rna_dynamic_decoder_mask
            ).sum(0)
        # all indices of non-zero GP genes
        net.features_idx_dict_["target_reconstructed_rna_idx"] = (
            torch.nonzero(self.target_n_gps_per_gene)).flatten().tolist()

        self.source_n_gps_per_gene = (
            source_rna_decoder_static_mask
            * net.source_rna_dynamic_decoder_mask
            ).sum(0)
        net.features_idx_dict_["source_reconstructed_rna_idx"] = (
            torch.nonzero(self.source_n_gps_per_gene)).flatten().tolist()

        self.target_rna_theta_reconstructed = net.target_rna_theta[
            net.features_idx_dict_["target_reconstructed_rna_idx"]].to(net.device)
        self.source_rna_theta_reconstructed = net.source_rna_theta[
            net.features_idx_dict_["source_reconstructed_rna_idx"]].to(net.device)    
        
        output = {}
        output["node_labels"] = {}

        # Get rna and atac part from omics feature vector
        # x_atac = x["atac"]
        # x_rna = x["rna"]
    
        # Compute aggregated neighborhood rna feature vector
        # rna_node_label_aggregator_output = self.rna_node_label_aggregator(
        #         x=x["rna"],
        #         edge_index=edge_index,
        #         return_agg_weights=return_agg_weights)
        # x_neighbors = rna_node_label_aggregator_output[0]

        # Retrieve rna node labels and only keep nodes in current node batch
        # and reconstructed features
        assert x["rna"].size(1) == net.n_output_genes_
        # assert x_neighbors.size(1) == self.n_output_genes_
        # 128 x 2999
        output["node_labels"]["target_rna"] = x["rna"][
            :, net.features_idx_dict_["target_reconstructed_rna_idx"]].to(net.device)
        # output["node_labels"]["source_rna"] = x_neighbors[batch_idx][
        #     :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
        
        # Use observed library size as scaling factor for the negative
        # binomial means of the rna distribution
        # 128 x 1
        target_rna_library_size = output["node_labels"]["target_rna"].sum(
            1).unsqueeze(1).to(net.device)
        # source_rna_library_size = output["node_labels"]["source_rna"].sum(
        #     1).unsqueeze(1)
        self.target_rna_log_library_size = torch.log(target_rna_library_size).to(net.device)
        # self.source_rna_log_library_size = torch.log(source_rna_library_size) 

        if len(net.keys) == 2:
            # Determine which features should be reconstructed based on
            # masks (if a feature is not connected to any node it should not
            # be reconstructed to not influence softmax activation outputs)
            if net.n_addon_gp_ > 0:
                target_atac_decoder_static_mask = torch.cat(
                    (net.target_atac_decoder_mask,
                        net.target_atac_decoder_addon_mask[0]), dim=0)
                source_atac_decoder_static_mask = torch.cat(
                    (net.source_atac_decoder_mask,
                        net.source_atac_decoder_addon_mask[0]), dim=0)
            else:
                target_atac_decoder_static_mask = net.target_atac_decoder_mask
                source_atac_decoder_static_mask = net.source_atac_decoder_mask

            self.target_n_gps_per_peak = (
                target_atac_decoder_static_mask
                * net.target_atac_dynamic_decoder_mask
                ).sum(0)
            net.features_idx_dict_["target_reconstructed_atac_idx"] = (
                torch.nonzero(self.target_n_gps_per_peak)).flatten().tolist()

            self.source_n_gps_per_peak = (
                source_atac_decoder_static_mask
                * net.source_atac_dynamic_decoder_mask
                ).sum(0)
            net.features_idx_dict_["source_reconstructed_atac_idx"] = (
                torch.nonzero(self.source_n_gps_per_peak)).flatten().tolist()

            self.target_atac_theta_reconstructed = net.target_atac_theta[
                net.features_idx_dict_["target_reconstructed_atac_idx"]].to(net.device)
            self.source_atac_theta_reconstructed = net.source_atac_theta[
                net.features_idx_dict_["source_reconstructed_atac_idx"]].to(net.device)

            # Compute aggregated neighborhood atac feature vector
            # atac_node_label_aggregator_output = (
            #     self.atac_node_label_aggregator(
            #         x=x_atac,
            #         edge_index=edge_index,
            #         return_agg_weights=return_agg_weights))
            # x_neighbors_atac = atac_node_label_aggregator_output[0]

            # Retrieve node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x["atac"].size(1) == net.n_output_peaks_
            # assert x_neighbors_atac.size(1) == self.n_output_peaks_
            # 128 x 4300
            output["node_labels"]["target_atac"] = x["atac"][
                :, net.features_idx_dict_["target_reconstructed_atac_idx"]].to(net.device) 
            # output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
            #     :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

            # Use observed library size as scaling factor for the negative
            # binomial means of the atac distribution
            # 128 x 1
            target_atac_library_size = output["node_labels"][
                "target_atac"].sum(1).unsqueeze(1).to(net.device)
            # source_atac_library_size = output["node_labels"][
            #     "source_atac"].sum(1).unsqueeze(1)
            self.target_atac_log_library_size = torch.log(
                target_atac_library_size).to(net.device)
            # self.source_atac_log_library_size = torch.log(
            #     source_atac_library_size)
            
            if update_atac_dynamic_decoder_mask:
                print("DONT WANT TO UPDATE ATAC DYNAMIC DECODER MASK!!!")
                # Get atac dynamic decoder masks to turn off peaks that
                # are mapped to only genes that are turned off
                with torch.no_grad():
                    # Retrieve rna decoder gp weights
                    gp_weights = net.get_gp_weights(
                        only_masked_features=False)[0].detach().cpu()
                    
                    # Round to 4 decimals as genes are never completely
                    # turned off due to L1 being not differentiable at 0
                    gp_weights = torch.round(gp_weights, decimals=4)

                    # Get boolean mask of non zero target and source gene 
                    # weights
                    non_zero_gene_weights = torch.ne(
                            gp_weights, 
                            0) # dim: (2 x n_genes, n_gps)
                    non_zero_target_gene_weights = non_zero_gene_weights[
                        :net.n_output_genes_, :] # dim: (n_genes, n_gps)
                    # non_zero_source_gene_weights = non_zero_gene_weights[
                    #     net.n_output_genes_:, :] # dim: (n_genes, n_gps)
                    
                    # Multiply boolean mask with gene peak mapping to remove
                    # peaks that are mapped to only turned off genes
                    target_atac_dynamic_decoder_mask = torch.mm(
                        non_zero_target_gene_weights.t().to(torch.float32), # dim: (n_gps,
                                                            #       n_genes)
                        net.gene_peaks_mask_.to(torch.float32)).to(torch.bool) # dim: (n_genes,
                                                # n_peaks)
                        # dim: (n_gps, n_peaks)
                    # source_atac_dynamic_decoder_mask = torch.mm(
                    #     non_zero_source_gene_weights.t().to(torch.float32),
                    #     net.gene_peaks_mask_.to(torch.float32)).to(torch.bool)
                    
                    # Create boolean mask of peaks (until here multiple
                    # active genes in a gp can be mapped to the same peak,
                    # resulting in values > 1.)
                    net.target_atac_dynamic_decoder_mask = (
                        net.target_atac_dynamic_decoder_mask & torch.ne(
                        target_atac_dynamic_decoder_mask, 
                        0)) # dim: (n_gps, n_peaks)
                    # net.source_atac_dynamic_decoder_mask = (
                    #     met.source_atac_dynamic_decoder_mask & torch.ne(
                    #     source_atac_dynamic_decoder_mask, 
                    #     0))
        
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        if dsc_only or epoch > self.n_epochs_rna:
            dsc_loss_flag = True
        else:
            dsc_loss_flag = False
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss}
         

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)


        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        x_nll, x_kl, x_elbo, x_elbo_flag = {}, {}, {}, {}
        for k in net.keys:
            if k == "rna":
                if rna_only or not atac_only:
                    target_rna_nb_means = net.u2x_targets[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], 
                        log_library_size=self.target_rna_log_library_size)[:, net.features_idx_dict_["target_reconstructed_rna_idx"]]
                    # source_rna_nb_means = net.u2x_sources[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], 
                    #     log_library_size=self.source_rna_log_library_size)[:, net.features_idx_dict_["source_reconstructed_rna_idx"]]
                    x_nll[k] = compute_omics_recon_nb_loss(
                            x=output["node_labels"]["target_rna"],
                            mu=target_rna_nb_means,
                            theta=torch.exp(self.target_rna_theta_reconstructed))
                    # x_nll[k] += (
                    #     lambda_gene_expr_recon * 
                    # compute_omics_recon_nb_loss(
                    #         x=output["node_labels"]["source_rna"],
                    #         mu=source_rna_nb_means,
                    #         theta=torch.exp(self.source_rna_theta_reconstructed)))
                    
                    # Compute KL divergence with epsilon added to avoid NaNs
                    x_kl[k] = D.kl_divergence(
                                u[k], prior
                            ).sum(dim=1).mean() / x[k].shape[1]
                    x_elbo[k] = x_nll[k] + self.lam_kl * x_kl[k]
                    x_elbo_flag[k] = True
                else:
                    x_nll[k] = torch.tensor(0.0, device=net.device)
                    x_kl[k] = torch.tensor(0.0, device=net.device)
                    x_elbo[k] = torch.tensor(0.0, device=net.device)
                    x_elbo_flag[k] = False

            elif k == "atac":
                if atac_only or not rna_only:
                    target_atac_nb_means = net.u2x_targets[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], 
                        log_library_size=self.target_atac_log_library_size,
                        dynamic_mask=net.target_atac_dynamic_decoder_mask)[:, net.features_idx_dict_["target_reconstructed_atac_idx"]]
                    # source_atac_nb_means = net.u2x_sources[k](usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], 
                    #     log_library_size=self.source_atac_log_library_size,
                    #     dynamic_mask=net.source_atac_dynamic_decoder_mask)[:, net.features_idx_dict_["source_reconstructed_atac_idx"]]    
                    # Compute target and source chromatin accessibility reconstruction
                    
                    # negative binomial loss for node batch
                    x_nll[k] = compute_omics_recon_nb_loss(
                            x=output["node_labels"]["target_atac"],
                            mu=target_atac_nb_means,
                            theta=torch.exp(self.target_atac_theta_reconstructed))
                    # x_nll[k] += (
                    #     lambda_chrom_access_recon * 
                    # compute_omics_recon_nb_loss(
                    #         x=output["node_labels"]["source_atac"],
                    #         mu=source_atac_nb_means,
                    #         theta=torch.exp(self.source_atac_theta_reconstructed)))
                    x_kl[k] = D.kl_divergence(
                                u[k], prior
                            ).sum(dim=1).mean() / x[k].shape[1]
                    x_elbo[k] = x_nll[k] + self.lam_kl * x_kl[k]
                    x_elbo_flag[k] = True
                else:
                    x_nll[k] = torch.tensor(0.0, device=net.device)
                    x_kl[k] = torch.tensor(0.0, device=net.device)
                    x_elbo[k] = torch.tensor(0.0, device=net.device)
                    x_elbo_flag[k] = False

        # x_nll_target = {
        #     k: -net.u2x_targets[k](
        #         usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }
        # x_nll_source = {
        #     k: -net.u2x_sources[k](
        #         usamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k], l[k]
        #     ).log_prob(x[k]).mean()
        #     for k in net.keys
        # }

        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys if x_elbo_flag[f"{k}"] is True)

        # Compute l1 reg loss of genes in masked gene programs
        masked_gp_l1_reg_loss = (self.lam_masked_l1 *
            compute_gp_l1_reg_loss(
                net,
                gp_type="prior",
                l1_targets_mask=self.l1_targets_mask,
                l1_sources_mask=self.l1_sources_mask))

        # Compute group lasso regularization loss of masked gene programs
        # group_lasso_reg_loss = (lambda_group_lasso *
        #     compute_gp_group_lasso_reg_loss(net))

        # Compute l1 regularization loss of genes in addon gene programs
        if net.n_addon_gp_ != 0:
            addon_gp_l1_reg_loss = (self.lam_addon_l1 *
            compute_gp_l1_reg_loss(net,
                                   gp_type="addon"))
        else:
            addon_gp_l1_reg_loss = 0.0

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss \
            + self.lam_masked_l1 * masked_gp_l1_reg_loss \
            + self.lam_addon_l1 * addon_gp_l1_reg_loss

        if not dsc_loss_flag:
            gen_loss = vae_loss
        else:
            gen_loss = vae_loss - self.lam_align * dsc_loss
        
        losses = {
            "dsc_loss": dsc_loss, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,"masked_gp_l1_loss": masked_gp_l1_reg_loss,
            "addon_gp_l1_loss": addon_gp_l1_reg_loss
        }
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss    
        
        return losses

    def set_requires_grad(self, modality, requires_grad):
        """
        Set the requires_grad for a specific modality or the discriminator.
        
        :param modality: Can be 'rna', 'atac', or 'dsc' (for discriminator)
        :param requires_grad: Boolean, whether to set requires_grad to True or False.
        """
        if modality == "rna":
            for param in self.net.x2u["rna"].parameters():
                param.requires_grad = requires_grad
            for param in self.net.u2x_targets["rna"].parameters():
                param.requires_grad = requires_grad
        elif modality == "atac":
            for param in self.net.x2u["atac"].parameters():
                param.requires_grad = requires_grad
            for param in self.net.u2x_targets["atac"].parameters():
                param.requires_grad = requires_grad
        elif modality == "dsc":
            for param in self.net.du.parameters():
                param.requires_grad = requires_grad

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch
        has_atac = "atac" in self.net.keys

        # use all gps initially, afterwards only use active gps
        if epoch < self.n_epochs_all_gp:
            self.use_only_active_gps = False
        else:
            self.use_only_active_gps = True
         

        # Freeze or unfreeze weights based on the current training phase
        if epoch <= self.n_epochs_rna:
            # Train only RNA weights
            self.set_requires_grad("rna", True)
            if has_atac:
                self.set_requires_grad("atac", False)
            self.set_requires_grad("dsc", False)
            rna_only = True
            atac_only = False
        elif has_atac and self.n_epochs_rna < epoch <= self.n_epochs_rna + self.n_epochs_atac:
            # Train only ATAC weights
            self.set_requires_grad("rna", False)
            self.set_requires_grad("atac", True)
            self.set_requires_grad("dsc", True)
            rna_only = False
            atac_only = True
        else:
            # Train both RNA and ATAC weights
            self.set_requires_grad("rna", True)
            if has_atac:
                self.set_requires_grad("atac", True)
            self.set_requires_grad("dsc", True)
            rna_only = False
            atac_only = False


        # Discriminator step
        # triple safety: only enter when need discriminator
        if not rna_only and epoch > self.n_epochs_rna:
            dsc_only = True
            if self.freeze_u:
                print("FREEZE_U WAS EVER TRUUUUEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                self.net.x2u.apply(freeze_running_stats)
                self.net.du.apply(freeze_running_stats)
            else:  # Discriminator step
                # Discriminator training happens only in the final phase
                losses = self.compute_losses(data, epoch, dsc_only=dsc_only, rna_only=rna_only, atac_only = atac_only, use_only_active_gps=self.use_only_active_gps)
                self.net.zero_grad(set_to_none=True)
                losses["dsc_loss"].backward()  # Already scaled by lam_align
                self.dsc_optim.step()
    
                # switch it back to (not only returning dsc_loss)
                dsc_only = False
        else:
            dsc_only = False

        # Generator step
        losses = self.compute_losses(data, epoch, dsc_only=dsc_only, rna_only=rna_only, atac_only=atac_only, use_only_active_gps=self.use_only_active_gps)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )


PairedDataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xtmp (data timepoint)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xadj (data spatial adjacency matrix)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # pmsk (paired mask)
    np.ndarray, # indices
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to PairedSTORMTrainer.compute_losses

@logged
class PairedSTORMTrainer(STORMTrainer):

    r"""
    Paired trainer for :class:`STORM`

    Parameters
    ----------
    net
        :class:`STORM` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    lam_cos
        Cosine similarity weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    def __init__(
            self, net: STORM, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None, lam_sup: float = None,
            lam_cos: float = None, lam_masked_l1: float = None, lam_addon_l1: float = None,
            lam_adj: float = None,
            lam_tmp: float = None,
            n_epochs_rna: int = None,
            n_epochs_atac: int = None,
            n_epochs_all_gp: int = None,
            normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align,
            lam_sup=lam_sup,
            lam_cos=lam_cos, lam_masked_l1=lam_masked_l1, lam_addon_l1=lam_addon_l1, 
            lam_adj=lam_adj, lam_tmp=lam_tmp, 
            n_epochs_rna = n_epochs_rna,
            n_epochs_atac = n_epochs_atac,
            n_epochs_all_gp = n_epochs_all_gp,
            normalize_u=normalize_u,
            modality_weight=modality_weight,
            optim=optim, lr=lr, **kwargs
        )
        required_kwargs = []
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        data_list = data[0]
    
        # Distribute the 13 tensors accordingly:
        x = data_list[0:K]          # First two tensors for `x`
        xrep = data_list[K:2*K]      # Next two tensors for `xrep`
        xbch = data_list[2*K:3*K]    # Next two tensors for `xbch`
        xtmp = data_list[3*K:4*K]    # Next two tensors for `xtmp`
        xlbl = data_list[4*K:5*K]    # Next two tensors for `xlbl`
        xdwt = data_list[5*K:6*K]    # Next two tensors for `xdwt`
        xadj = data_list[6*K:7*K]    # The next tensors for `xadj`
        
        pmsk = data_list[7*K]        # The 13th tensor is `pmsk`

        # print(f"Indices content and type: {data[1]} of type {data[1].dtype}")
        # numpy array
        indices = data[1]

        # Handle the graph data: eidx, ewt, esgn (elements 2, 3, and 4)
        eidx = data[2]
        ewt = data[3]
        esgn = data[4]

        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xtmp = {
            k: xtmp[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xadj = {
            k: xadj[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.as_tensor(
                i, dtype=torch.int64, device=device
            ).expand(x[k].shape[0])
            for i, k in enumerate(keys)
        }
        pmsk = pmsk.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xtmp, xlbl, xdwt, xflag, xadj, pmsk, indices, eidx, ewt, esgn

    def compute_losses(
           self, data: PairedDataTensors, epoch: int, dsc_only: bool = False,
           rna_only: bool = False, atac_only: bool = False,
           use_only_active_gps: bool = False, return_agg_weights: bool=False,
           update_atac_dynamic_decoder_mask: bool=False
    ) -> Mapping[str, torch.Tensor]:
        net = self.net
        # eidx: 2 x 22092
        x, xrep, xbch, xtmp, xlbl, xdwt, xflag, xadj, pmsk, indicies, eidx, ewt, esgn = data
        # adj matricies
        # print(f"xadj for epoch {epoch} is {xadj}")
        # cell names
        # print(f"indicies look like {indicies}")

        u, l = {}, {}
        for k in net.keys:
            u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        # w['rna/atac'] = Normal(loc= 128 * 50, scale = 128 * 50)
        # l['rna/atac'] = None
        # usamp['rna/atac'].shape = 128 * 50
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            # 128 x 50
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}


        zsamp = {}
        for k in net.keys:
            # 128 x 178
            zsamp[k] = net.u2z[k](usamp[k])
        
        if use_only_active_gps:
            active_gp_mask = net.get_active_gp_mask()
            # Set gp scores of inactive gene programs to 0 to not affect
            # graph decoder
            zsamp["rna"] = zsamp["rna"].clone()
            zsamp["rna"][:, ~active_gp_mask] = 0
            if "atac" in net.keys:
                active_gp_mask_atac = net.get_active_gp_mask(which_modality="atac")
                zsamp["atac"] = zsamp["atac"].clone()
                zsamp["atac"][:, ~active_gp_mask_atac] = 0

        # 2 x 128
        pmsk = pmsk.T
        assert (torch.all(pmsk))
        # [2, 128, N_gp]
        # zsamp_stack = torch.stack([zsamp[k] for k in net.keys])
        # [2, 128, 50]
        usamp_stack = torch.stack([usamp[k] for k in net.keys])
        # [2, 128, 50]
        pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        # [128, N_gp]
        # zsamp_mean = (zsamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        # usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        # if use_only_active_gps:
        #     active_gp_mask = net.get_active_gp_mask()
        #     zsamp_mean[:, ~active_gp_mask] = 0
        # if self.normalize_u:
        #     usamp_mean = F.normalize(usamp_mean, dim=1)

        # Normal(loc: 7299 x 50, scale: 7299 x 50)
        v = net.g2v(self.eidx, self.enorm, self.esgn)
        # 7299 x 50
        vsamp = v.rsample()

        with torch.no_grad():
            if net.training:
                # Update running mean absolute gp scores using exponential
                # moving average with momentum of 0.1
                # 50
                mean_abs_mu = u["rna"].mean.norm(p=1, dim=0) / u["rna"].mean.size(0)
                # 178
                mean_abs_mu = net.u2z["rna"](mean_abs_mu)
                # running_mean_abs_mu = N_gp = 178
                net.running_mean_abs_mu = (
                    0.1 * mean_abs_mu + 0.9 * net.running_mean_abs_mu.to(net.device))

                if "atac" in net.keys:
                    mean_abs_mu = u["atac"].mean.norm(p=1, dim=0) / u["atac"].mean.size(0)
                    # 178
                    mean_abs_mu = net.u2z["atac"](mean_abs_mu)
                    # running_mean_abs_mu = N_gp = 178
                    net.running_mean_abs_mu_atac = (
                        0.1 * mean_abs_mu + 0.9 * net.running_mean_abs_mu_atac.to(net.device))
            if use_only_active_gps:
                # Set running mean abs mu of inactive gene programs to 0 for
                # active gp determination
                net.running_mean_abs_mu[~active_gp_mask] = 0
                if "atac" in net.keys:
                    net.running_mean_abs_mu_atac[~active_gp_mask_atac] = 0

                # Set dynamic mask to 0 for all inactive gene programs to
                # not affect omics decoders
                net.target_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0
                net.source_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0

                if "atac" in net.keys:
                    net.target_atac_dynamic_decoder_mask[~active_gp_mask_atac, :] = 0
                    net.source_atac_dynamic_decoder_mask[~active_gp_mask_atac, :] = 0

        # Determine which features should be reconstructed based on
        # static and dynamic masks (if a feature is not connected to any
        # node it should not be reconstructed to not influence softmax
        # activation outputs). This can happen when no add-on gene programs
        # are present or when gene programs are turned off.
        if net.n_addon_gp_ > 0:
            # N_gp x N_genes
            target_rna_decoder_static_mask = torch.cat(
                (net.target_rna_decoder_mask,
                    net.target_rna_decoder_addon_mask[0]), dim=0)
            # N_gp x N_peaks
            source_rna_decoder_static_mask = torch.cat(
                (net.source_rna_decoder_mask,
                    net.source_rna_decoder_addon_mask[0]), dim=0)
        else:
            target_rna_decoder_static_mask = net.target_rna_decoder_mask
            source_rna_decoder_static_mask = net.source_rna_decoder_mask
        # N_genes
        self.target_n_gps_per_gene = (
            target_rna_decoder_static_mask
            * net.target_rna_dynamic_decoder_mask
            ).sum(0)
        # all indices of non-zero GP genes
        # N target genes that are in at least 1 GP = N_genes
        net.features_idx_dict_["target_reconstructed_rna_idx"] = (
            torch.nonzero(self.target_n_gps_per_gene)).flatten().tolist()
        
        # N_genes
        self.source_n_gps_per_gene = (
            source_rna_decoder_static_mask
            * net.source_rna_dynamic_decoder_mask
            ).sum(0)
        # N source genes that are in at least 1 GP (including addons) = N_genes
        net.features_idx_dict_["source_reconstructed_rna_idx"] = (
            torch.nonzero(self.source_n_gps_per_gene)).flatten().tolist()


        self.target_rna_theta_reconstructed = net.target_rna_theta[
            net.features_idx_dict_["target_reconstructed_rna_idx"]].to(net.device)
        self.source_rna_theta_reconstructed = net.source_rna_theta[
            net.features_idx_dict_["source_reconstructed_rna_idx"]].to(net.device)   
        
        output = {}
        output["node_labels"] = {}

    
        # Compute aggregated neighborhood rna feature vector
        x_neighbors = net.node_label_aggregator["rna"](
                x=x["rna"], xadj = xadj["rna"])

        # Retrieve rna node labels and only keep nodes in current node batch
        # and reconstructed features
        assert x["rna"].size(1) == net.n_output_genes_
        assert x_neighbors.size(1) == net.n_output_genes_
        # 128 x 2999 select all N_genes = batchsize x N_genes
        output["node_labels"]["target_rna"] = x["rna"][
            :, net.features_idx_dict_["target_reconstructed_rna_idx"]].to(net.device)
        output["node_labels"]["source_rna"] = x_neighbors[
            :, net.features_idx_dict_["source_reconstructed_rna_idx"]].to(net.device)
        
        # Use observed library size as scaling factor for the negative
        # binomial means of the rna distribution
        # sum expression across N_genes for each sample = 128 x 1
        target_rna_library_size = output["node_labels"]["target_rna"].sum(
            1).unsqueeze(1).to(net.device)
        source_rna_library_size = output["node_labels"]["source_rna"].sum(
            1).unsqueeze(1)
        self.target_rna_log_library_size = torch.log(target_rna_library_size).to(net.device)
        self.source_rna_log_library_size = torch.log(source_rna_library_size).to(net.device)

        if "atac" in net.keys:
            # Determine which features should be reconstructed based on
            # masks (if a feature is not connected to any node it should not
            # be reconstructed to not influence softmax activation outputs)
            if net.n_addon_gp_ > 0:
                target_atac_decoder_static_mask = torch.cat(
                    (net.target_atac_decoder_mask,
                        net.target_atac_decoder_addon_mask[0]), dim=0)
                source_atac_decoder_static_mask = torch.cat(
                    (net.source_atac_decoder_mask,
                        net.source_atac_decoder_addon_mask[0]), dim=0)
            else:
                target_atac_decoder_static_mask = net.target_atac_decoder_mask
                source_atac_decoder_static_mask = net.source_atac_decoder_mask


            self.target_n_gps_per_peak = (
                target_atac_decoder_static_mask
                * net.target_atac_dynamic_decoder_mask
                ).sum(0)
            net.features_idx_dict_["target_reconstructed_atac_idx"] = (
                torch.nonzero(self.target_n_gps_per_peak)).flatten().tolist()


            self.source_n_gps_per_peak = (
                source_atac_decoder_static_mask
                * net.source_atac_dynamic_decoder_mask
                ).sum(0)
            net.features_idx_dict_["source_reconstructed_atac_idx"] = (
                torch.nonzero(self.source_n_gps_per_peak)).flatten().tolist()


            self.target_atac_theta_reconstructed = net.target_atac_theta[
                net.features_idx_dict_["target_reconstructed_atac_idx"]].to(net.device)
            self.source_atac_theta_reconstructed = net.source_atac_theta[
                net.features_idx_dict_["source_reconstructed_atac_idx"]].to(net.device)


            # Compute aggregated neighborhood atac feature vector
            x_neighbors_atac = net.node_label_aggregator["atac"](
                x=x["atac"], xadj = xadj["atac"])

            # Retrieve node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x["atac"].size(1) == net.n_output_peaks_
            assert x_neighbors_atac.size(1) == net.n_output_peaks_
            # 128 x 4300
            output["node_labels"]["target_atac"] = x["atac"][
                :, net.features_idx_dict_["target_reconstructed_atac_idx"]].to(net.device)
            output["node_labels"]["source_atac"] = x_neighbors_atac[
                :, net.features_idx_dict_["source_reconstructed_atac_idx"]].to(net.device)


            # Use observed library size as scaling factor for the negative
            # binomial means of the atac distribution
            # 128 x 1
            target_atac_library_size = output["node_labels"][
                "target_atac"].sum(1).unsqueeze(1).to(net.device)
            source_atac_library_size = output["node_labels"][
                "source_atac"].sum(1).unsqueeze(1).to(net.device)
            self.target_atac_log_library_size = torch.log(
                target_atac_library_size).to(net.device)
            self.source_atac_log_library_size = torch.log(
                source_atac_library_size).to(net.device)
            
            # if update_atac_dynamic_decoder_mask:
            #     print("DONT WANT TO UPDATE ATAC DYNAMIC DECODER MASK!!!")
            #     # Get atac dynamic decoder masks to turn off peaks that
            #     # are mapped to only genes that are turned off
            #     with torch.no_grad():
            #         # Retrieve rna decoder gp weights
            #         gp_weights = net.get_gp_weights(
            #             only_masked_features=False)[0].detach().cpu()
                    
            #         # Round to 4 decimals as genes are never completely
            #         # turned off due to L1 being not differentiable at 0
            #         gp_weights = torch.round(gp_weights, decimals=4)


            #         # Get boolean mask of non zero target and source gene
            #         # weights
            #         non_zero_gene_weights = torch.ne(
            #                 gp_weights,
            #                 0) # dim: (2 x n_genes, n_gps)
            #         non_zero_target_gene_weights = non_zero_gene_weights[
            #             :net.n_output_genes_, :] # dim: (n_genes, n_gps)
            #         # non_zero_source_gene_weights = non_zero_gene_weights[
            #         #     net.n_output_genes_:, :] # dim: (n_genes, n_gps)
                    
            #         # Multiply boolean mask with gene peak mapping to remove
            #         # peaks that are mapped to only turned off genes
            #         target_atac_dynamic_decoder_mask = torch.mm(
            #             non_zero_target_gene_weights.t().to(torch.float32), # dim: (n_gps,
            #                                                 #       n_genes)
            #             net.gene_peaks_mask_.to(torch.float32)).to(torch.bool) # dim: (n_genes,
            #                                     # n_peaks)
            #             # dim: (n_gps, n_peaks)
            #         # source_atac_dynamic_decoder_mask = torch.mm(
            #         #     non_zero_source_gene_weights.t().to(torch.float32),
            #         #     net.gene_peaks_mask_.to(torch.float32)).to(torch.bool)
                    
            #         # Create boolean mask of peaks (until here multiple
            #         # active genes in a gp can be mapped to the same peak,
            #         # resulting in values > 1.)
            #         net.target_atac_dynamic_decoder_mask = (
            #             net.target_atac_dynamic_decoder_mask & torch.ne(
            #             target_atac_dynamic_decoder_mask,
            #             0)) # dim: (n_gps, n_peaks)
            #         # net.source_atac_dynamic_decoder_mask = (
            #         #     met.source_atac_dynamic_decoder_mask & torch.ne(
            #         #     source_atac_dynamic_decoder_mask,
            #         #     0))

        # Normal(loc = 0.0, scale = 1.0)
        prior = net.prior()

        # 256 x 50
        u_cat = torch.cat([u[k].mean for k in net.keys])
        # 256
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        # 256
        xtmp_cat = torch.cat([xtmp[k] for k in net.keys])
        # [2, 128]
        xbch_stack = torch.stack([xbch[k] for k in net.keys])
        # 256
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        # 256
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = D.Normal(0, u_cat.std(axis=0)).sample((u_cat.shape[0], ))
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise

        modality_logits, timepoint_logits = net.du(u_cat)
        timepoint_loss = F.cross_entropy(timepoint_logits, xtmp_cat, reduction="none")
        timepoint_loss = (timepoint_loss * xdwt_cat).sum() / xdwt_cat.numel()
        dsc_loss_t = self.lam_tmp * timepoint_loss
        if len(net.keys) > 1:
            modality_loss = F.cross_entropy(modality_logits, xflag_cat, reduction="none")
            modality_loss = (modality_loss * xdwt_cat).sum() / xdwt_cat.numel()
            dsc_loss_m = self.lam_align * modality_loss 
            dsc_loss = dsc_loss_m  + dsc_loss_t
        else:
            dsc_loss_m = torch.as_tensor(0.0, device=net.device)
            dsc_loss = dsc_loss_t

        if dsc_only or epoch > self.n_epochs_rna:
            # only calculate gen_loss as vae_loss - dsc_loss when flag true; or else dsc loss grad error
            dsc_loss_flag = True
        else:
            dsc_loss_flag = False
        if dsc_only:
            return {"dsc_loss": dsc_loss}
        
        # don't need the concatenated versions
        xbch_cat, xdwt_cat, xflag_cat, anneal = None, None, None, None


        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.tensor(0.0, device=self.net.device)


        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl


        x_nll, x_kl, x_elbo, x_elbo_flag, x_nll_log, x_kl_log, x_elbo_log = {}, {}, {}, {}, {}, {}, {}
        for k in net.keys:
            if k == "rna":
                target_rna_nb_means = net.u2x_targets[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k],
                    log_library_size=self.target_rna_log_library_size)[:, net.features_idx_dict_["target_reconstructed_rna_idx"]]
                source_rna_nb_means = net.u2x_sources[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k],
                    log_library_size=self.source_rna_log_library_size)[:, net.features_idx_dict_["source_reconstructed_rna_idx"]]
                
                x_nll_log[k] = compute_omics_recon_nb_loss(
                        x=output["node_labels"]["target_rna"],
                        mu=target_rna_nb_means,
                        theta=torch.exp(self.target_rna_theta_reconstructed))
                x_nll_log[k] += compute_omics_recon_nb_loss(
                        x=output["node_labels"]["source_rna"],
                        mu=source_rna_nb_means,
                        theta=torch.exp(self.source_rna_theta_reconstructed))

                x_kl_log[k] = D.kl_divergence(
                            u[k], prior
                        ).sum(dim=1).mean() / x[k].shape[1]
                x_elbo_log[k] = x_nll_log[k] + self.lam_kl * x_kl_log[k]
                x_elbo_flag[k] = True

                if rna_only or not atac_only:
                    x_nll[k] = x_nll_log[k]
                    x_kl[k] = x_kl_log[k]
                    x_elbo[k] = x_elbo_log[k]
                else:
                    # zero out rna loss during backprop when not training RNA, still reports the loss though
                    x_nll[k] = 0.0 * x_nll_log[k]
                    x_kl[k] = 0.0 * x_kl_log[k]
                    x_elbo[k] = 0.0 * x_elbo_log[k]
                    x_elbo_flag[k] = False

            elif k == "atac":
                target_atac_nb_means = net.u2x_targets[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k],
                    log_library_size=self.target_atac_log_library_size,
                    dynamic_mask=net.target_atac_dynamic_decoder_mask)[:, net.features_idx_dict_["target_reconstructed_atac_idx"]]
                source_atac_nb_means = net.u2x_sources[k](zsamp[k], vsamp[getattr(net, f"{k}_idx")], xbch[k],
                    log_library_size=self.source_atac_log_library_size,
                    dynamic_mask=net.source_atac_dynamic_decoder_mask)[:, net.features_idx_dict_["source_reconstructed_atac_idx"]]   
                
                # negative binomial loss for node batch
                x_nll_log[k] = compute_omics_recon_nb_loss(
                        x=output["node_labels"]["target_atac"],
                        mu=target_atac_nb_means,
                        theta=torch.exp(self.target_atac_theta_reconstructed))
                x_nll_log[k] += compute_omics_recon_nb_loss(
                        x=output["node_labels"]["source_atac"],
                        mu=source_atac_nb_means,
                        theta=torch.exp(self.source_atac_theta_reconstructed))

                x_kl_log[k] = D.kl_divergence(
                            u[k], prior
                        ).sum(dim=1).mean() / x[k].shape[1]
                x_elbo_log[k] = x_nll_log[k] + self.lam_kl * x_kl_log[k]
                x_elbo_flag[k] = True

                if atac_only or not rna_only:
                    x_nll[k] = x_nll_log[k]
                    x_kl[k] = x_kl_log[k]
                    x_elbo[k] = x_elbo_log[k]
                else:
                    # zero out atac loss during backprop when not training ATAC, still reports the loss though
                    x_nll[k] = 0.0 * x_nll_log[k]
                    x_kl[k] = 0.0 * x_kl_log[k]
                    x_elbo[k] = 0.0 * x_elbo_log[k]
                    x_elbo_flag[k] = False
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys if x_elbo_flag[f"{k}"] is True)


        # Compute l1 reg loss of genes in masked gene programs
        masked_gp_l1_reg_loss = (self.lam_masked_l1 *
            compute_gp_l1_reg_loss(
                net,
                gp_type="prior",
                l1_targets_mask=self.l1_targets_mask,
                l1_sources_mask=self.l1_sources_mask))


        # Compute l1 regularization loss of genes in addon gene programs
        if net.n_addon_gp_ != 0:
            addon_gp_l1_reg_loss = (self.lam_addon_l1 *
            compute_gp_l1_reg_loss(net,
                                    gp_type="addon"))

        # TODO: rename cos loss to MSE
        if not rna_only and "atac" in net.keys:
            cos_loss = 0.0
            for i, m in enumerate(pmsk):
                if m.sum():
                    if i == 0:
                        cos_loss += torch.mean((usamp_stack[i, m] - usamp_stack[1, m]) ** 2)
                    elif i == 1:
                        cos_loss += torch.mean((usamp_stack[i, m] - usamp_stack[0, m]) ** 2)
        else:
            cos_loss = torch.as_tensor(0.0, device=net.device)
        

        if self.lam_adj:
            adj_loss = 0.0
            for k in net.keys:
                if k == "rna":
                    if rna_only or not atac_only:
                        cosine_similarity_matrix = F.cosine_similarity(zsamp[k].unsqueeze(1), zsamp[k].unsqueeze(0), dim=-1)
                        # Compute loss using negative sampling
                        loss = negative_sampling_loss(cosine_similarity_matrix, xadj[k])
                        # print(f"Loss: {loss.item()}")
                        adj_loss += torch.as_tensor(loss, device=net.device)

                elif k == "atac":
                    if atac_only or not rna_only:
                        cosine_similarity_matrix = F.cosine_similarity(zsamp[k].unsqueeze(1), zsamp[k].unsqueeze(0), dim=-1)
                        # Compute loss using negative sampling
                        loss = negative_sampling_loss(cosine_similarity_matrix, xadj[k])
                        # print(f"Loss: {loss.item()}")
                        adj_loss += torch.as_tensor(loss, device=net.device)
                
        else:
            adj_loss =  torch.as_tensor(0.0, device=net.device)

        
        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss \
            + self.lam_masked_l1 * masked_gp_l1_reg_loss \
            + self.lam_adj * adj_loss 
        
        if "atac" in net.keys:
            vae_loss += self.lam_cos * cos_loss

        if net.n_addon_gp_ != 0:
            vae_loss += self.lam_addon_l1 * addon_gp_l1_reg_loss
        
        if not dsc_loss_flag:
            gen_loss = vae_loss
        else:
            gen_loss = vae_loss - dsc_loss 
        
        losses = {
            "dsc_loss": dsc_loss_m, "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            "cos_loss": cos_loss,
            "masked_gp_l1_loss": masked_gp_l1_reg_loss,
            "adj_loss": adj_loss,
            "dsc_loss_t": dsc_loss_t
        }
        if net.n_addon_gp_ != 0:
            losses["addon_gp_l1_loss"] = addon_gp_l1_reg_loss

        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll_log[k],
                f"x_{k}_kl": x_kl_log[k],
                f"x_{k}_elbo": x_elbo_log[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

        

#--------------------------------- Public API ----------------------------------

# Default AnnData keys consumed by STORMModel during training. Users may
# override any subset via the ``anndata_keys`` constructor argument; the dict
# below also enumerates every logical key the model knows about, so unknown
# keys passed by the user are rejected as typos.
STORM_DEFAULT_KEYS: Mapping[str, str] = {
    # rna.varm
    "gp_targets_mask": "storm_gp_targets",
    "gp_sources_mask": "storm_gp_sources",
    "gene_peaks_mask": "storm_gene_peaks",
    "gp_targets_categories_mask": "storm_gp_targets_categories",
    "gp_sources_categories_mask": "storm_gp_sources_categories",
    # atac.varm
    "ca_targets_mask": "storm_ca_targets",
    "ca_sources_mask": "storm_ca_sources",
    # rna.uns
    "genes_idx": "storm_genes_idx",
    "target_genes_idx": "storm_target_genes_idx",
    "source_genes_idx": "storm_source_genes_idx",
    "gp_names": "storm_gp_names",
    "targets_categories_label_encoder": "storm_targets_categories_label_encoder",
    "sources_categories_label_encoder": "storm_sources_categories_label_encoder",
    # atac.uns
    "peaks_idx": "storm_peaks_idx",
    "target_peaks_idx": "storm_target_peaks_idx",
    "source_peaks_idx": "storm_source_peaks_idx",
}


def _resolve_anndata_keys(
        anndata_keys: Optional[Mapping[str, str]]
) -> Mapping[str, str]:
    r"""Merge user-supplied AnnData key overrides on top of STORM_DEFAULT_KEYS."""
    if not anndata_keys:
        return dict(STORM_DEFAULT_KEYS)
    unknown = set(anndata_keys) - set(STORM_DEFAULT_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown anndata_keys entries: {sorted(unknown)}. "
            f"Valid keys are: {sorted(STORM_DEFAULT_KEYS)}."
        )
    resolved = dict(STORM_DEFAULT_KEYS)
    resolved.update(anndata_keys)
    return resolved


@logged
class STORMModel(Model, nn.Module):

    r"""
    GLUE model for single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    NET_TYPE = STORM
    TRAINER_TYPE = STORMTrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, AnnData],
            vertices: List[str], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            random_seed: int = 0,
            n_addon_gp: int=0,
            active_gp_type: Literal["mixed", "separate"]="separate",
            active_gp_thresh_ratio: float=0.0,
            dropout_rate_graph_decoder: float=0.,
            include_edge_recon_loss: bool=True,
            include_edge_kl_loss: bool=True,
            anndata_keys: Optional[Mapping[str, str]] = None,
    ) -> None:
        # Resolve AnnData key overrides (e.g. {"gene_peaks_mask": "my_gene_peaks"}).
        # See STORM_DEFAULT_KEYS for the full list of overridable logical names.
        self.anndata_keys_ = _resolve_anndata_keys(anndata_keys)

        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_edge_kl_loss_ = include_edge_kl_loss
        
        # Retrieve gene program masks
        gp_targets_mask_key = self.anndata_keys_["gp_targets_mask"]
        if gp_targets_mask_key in adatas["rna"].varm:
            # NOTE: dtype can be changed to bool and should be able to handle sparse
            # mask
            self.gp_targets_mask_ = torch.tensor(
                adatas["rna"].varm[gp_targets_mask_key].T,
                dtype=torch.bool)
        else:
            raise ValueError("Please specify an adequate ´gp_targets_mask_key´ "
                             "for your adata object. The targets mask needs to "
                             "be stored in ´adata.varm[gp_targets_mask_key]´. "
                             " If you do not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s that"
                             " allows all gene program latent nodes to "
                             "reconstruct all genes.")
        
        gp_sources_mask_key = self.anndata_keys_["gp_sources_mask"]
        if gp_sources_mask_key in adatas["rna"].varm:
            # NOTE: dtype can be changed to bool and should be able to handle
            # sparse mask
            self.gp_sources_mask_ = torch.tensor(
                adatas["rna"].varm[gp_sources_mask_key].T,
                dtype=torch.bool)
                                           
        else:
            raise ValueError("Please specify an adequate "
                             "´gp_sources_mask_key´ for your adata object. "
                             "The sources mask needs to be stored in "
                             "´adata.varm[gp_sources_mask_key]´. If you do "
                             "not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s "
                             " that allows all gene program latent nodes to"
                             " reconstruct all genes.")

        if len(adatas) == 1:
            self.ca_targets_mask_ = None
            self.ca_sources_mask_ = None
            gene_peaks_mask = None
        else:
            # turn into COO format
            gene_peaks_mask_key = self.anndata_keys_["gene_peaks_mask"]
            gene_peaks_mask = adatas["rna"].varm[gene_peaks_mask_key].tocoo()
            gene_peaks_mask = torch.sparse_coo_tensor(
                indices=[gene_peaks_mask.row, gene_peaks_mask.col],
                values=gene_peaks_mask.data,
                size=gene_peaks_mask.shape,
                dtype=torch.bool) # bool does not work with torch.mm
            ca_targets_mask_key = self.anndata_keys_["ca_targets_mask"]
            if ca_targets_mask_key in adatas["atac"].varm:
                ca_targets_mask = adatas["atac"].varm[ca_targets_mask_key].T.tocoo()
            else:
                raise ValueError("Please specify an adequate "
                                 "´ca_targets_mask_key´ for your adata_atac "
                                 "object. The targets mask needs to be stored "
                                 "in ´adata_atac.varm[ca_targets_mask_key]´. If"
                                 " you do not want to mask chromatin "
                                 " accessibility reconstruction, you can create"
                                 " a mask of 1s that allows all gene program "
                                 "latent nodes to reconstruct all peaks.")
            
            # 1898 x 4283
            self.ca_targets_mask_ = torch.sparse_coo_tensor(
                indices=[ca_targets_mask.row, ca_targets_mask.col],
                values=ca_targets_mask.data,
                size=ca_targets_mask.shape,
                dtype=torch.bool).to_dense() # for now
            ca_sources_mask_key = self.anndata_keys_["ca_sources_mask"]
            if ca_sources_mask_key in adatas["atac"].varm:
                ca_sources_mask = adatas["atac"].varm[
                    ca_sources_mask_key].T.tocoo()
                # 1898 x 4283
                self.ca_sources_mask_ = torch.sparse_coo_tensor(
                    indices=[ca_sources_mask.row, ca_sources_mask.col],
                    values=ca_sources_mask.data,
                    size=ca_sources_mask.shape,
                    dtype=torch.bool).to_dense() # for now
            else:
                raise ValueError("Please specify an adequate "
                                "´ca_sources_mask_key´ for your adata_atac "
                                "object. The sources mask needs to be "
                                "stored in "
                                "´adata_atac.varm[ca_sources_mask_key]´. If"
                                "you do not want to mask chromatin "
                                " accessibility reconstruction, you can "
                                "create a mask of 1s that allows all gene "
                                "program latent nodes to reconstruct all "
                                "peaks.")

        # Retrieve index of genes in gp mask and index of genes not in gp mask
        self.features_idx_dict_ = {}
        genes_idx_key = self.anndata_keys_["genes_idx"]
        self.features_idx_dict_["masked_rna_idx"] = adatas["rna"].uns[
            genes_idx_key]
        self.features_idx_dict_["unmasked_rna_idx"] = [
            i for i in range(len(adatas["rna"].var_names))
            if i not in self.features_idx_dict_["masked_rna_idx"]]
        target_genes_idx_key = self.anndata_keys_["target_genes_idx"]
        self.features_idx_dict_["target_masked_rna_idx"] = list(
            adatas["rna"].uns[target_genes_idx_key])
        self.features_idx_dict_["target_unmasked_rna_idx"] = [
            i for i in range(len(adatas["rna"].var_names))
            if i not in self.features_idx_dict_["target_masked_rna_idx"]]
        source_genes_idx_key = self.anndata_keys_["source_genes_idx"]
        self.features_idx_dict_["source_masked_rna_idx"] = list(
            adatas["rna"].uns[source_genes_idx_key])
        self.features_idx_dict_["source_unmasked_rna_idx"] = [
            i for i in range(len(adatas["rna"].var_names))
            if i not in self.features_idx_dict_["source_masked_rna_idx"]]
        
        # Retrieve index of peaks in ca mask and index of peaks not in ca mask
        if len(adatas) > 1:
            peaks_idx_key = self.anndata_keys_["peaks_idx"]
            target_peaks_idx_key = self.anndata_keys_["target_peaks_idx"]
            source_peaks_idx_key = self.anndata_keys_["source_peaks_idx"]
            self.peaks_idx_ = adatas["atac"].uns[peaks_idx_key]
            self.target_peaks_idx_ = adatas["atac"].uns[target_peaks_idx_key]
            self.source_peaks_idx_ = adatas["atac"].uns[source_peaks_idx_key]
            
            self.features_idx_dict_["masked_atac_idx"] = adatas["atac"].uns[
                peaks_idx_key]
            self.features_idx_dict_["unmasked_atac_idx"] = [
                i for i in range(len(adatas["atac"].var_names))
                if i not in self.features_idx_dict_["masked_atac_idx"]]
            self.features_idx_dict_["target_masked_atac_idx"] = list(
                adatas["atac"].uns[target_peaks_idx_key])
            self.features_idx_dict_["target_unmasked_atac_idx"] = [
                i for i in range(len(adatas["atac"].var_names))
                if i not in self.features_idx_dict_["target_masked_atac_idx"]]
            self.features_idx_dict_["source_masked_atac_idx"] = list(
                adatas["atac"].uns[source_peaks_idx_key])
            self.features_idx_dict_["source_unmasked_atac_idx"] = [
                i for i in range(len(adatas["atac"].var_names))
                if i not in self.features_idx_dict_["source_masked_atac_idx"]]
        
        self.n_input_ = adatas["rna"].n_vars
        self.n_output_genes_ = adatas["rna"].n_vars
        if len(adatas) > 1:
            self.modalities_ = ["rna", "atac"]
            if not np.all(adatas["rna"].obs.index == adatas["atac"].obs.index):
                raise ValueError("Please make sure that 'adata' and "
                                 "'adata_atac' contain the same observations in"
                                 " the same order.")
            # Peaks are concatenated to genes in input <- NOT TRUE IN OUR CASE
            self.n_input_ += adatas["atac"].n_vars
            self.n_output_peaks_ = adatas["atac"].n_vars
        else:
            self.modalities_ = ["rna"]
            self.n_output_peaks_ = 0

        self.n_prior_gp_ = len(self.gp_targets_mask_)
        self.n_addon_gp_ = n_addon_gp

        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder


        gp_names_key = self.anndata_keys_["gp_names"]
        if n_addon_gp > 0:
            # Add add-on gps to adata
            gp_list = list(adatas["rna"].uns[gp_names_key])
            for i in range(n_addon_gp):
                if f"Add-on_{i}_GP" not in gp_list:
                    gp_list.append(f"Add-on_{i}_GP")
            adatas["rna"].uns[gp_names_key] = np.array(gp_list)
        else:
            # Remove add-on gps from adata
            for gp_name in list(adatas["rna"].uns[gp_names_key]):
                if "Add-on" in gp_name:
                    adatas["rna"].uns[gp_names_key] = np.delete(
                        adatas["rna"].uns[gp_names_key],
                        list(adatas["rna"].uns[gp_names_key]).index(gp_name))
        
        # seperate
        # Type to determine active gene programs. Can be ´mixed´, in which case
        # active gene programs are determined across prior and add-on gene programs
        # jointly or ´separate´ in which case they are determined separately for
        # prior adn add-on gene programs.
        self.active_gp_type_ = active_gp_type

         # Determine features scale factors = sum of first column
        # TODO: .X or counts?
        counts = adatas["rna"].layers["counts"]
        # Check if the matrix is sparse
        if issparse(counts):
            # Sum along the first axis for sparse matrices and convert to a dense array
            summed_counts = counts.sum(axis=0).A1  # Use `.A1` to get a flat array from sparse
        else:
            # Sum along the first axis for dense arrays
            summed_counts = counts.sum(axis=0)
        tensor_counts = torch.tensor(summed_counts)
        self.features_scale_factors_ = torch.concat((tensor_counts, tensor_counts))

        # Determine dimensionality of hidden encoder layer if not provided
        if len(adatas["rna"].var) > (self.n_prior_gp_ + self.n_addon_gp_):
            n_hidden_encoder = (self.n_prior_gp_ + self.n_addon_gp_)
        else:
            n_hidden_encoder = len(adatas["rna"].var)
        self.n_hidden_encoder_ = n_hidden_encoder


        graph_decoder = sc.CosineSimGraphDecoder(
            dropout_rate=self.dropout_rate_graph_decoder_)

        for entity in ["target", "source"]:
            if n_addon_gp > 0:
                # Initialize rna add-on masks which are 0 everywhere except
                # for the genes that are unmasked, in which case they are 1
                rna_decoder_addon_mask = torch.zeros(
                    n_addon_gp,
                    self.n_output_genes_,
                    dtype=torch.float32)
                rna_decoder_addon_mask[
                    :, self.features_idx_dict_[f"{entity}_unmasked_rna_idx"]] = 1.
                setattr(self,
                        f"{entity}_rna_decoder_addon_mask",
                        rna_decoder_addon_mask)
                
                # Set add-on rna idx to unmasked rna idx as all unmasked
                # genes are part of add-on gps
                self.features_idx_dict_[f"{entity}_addon_rna_idx"] = (
                    self.features_idx_dict_[f"{entity}_unmasked_rna_idx"])
                    
                if "atac" in self.modalities_:
                    # Initialize atac add-on masks which are 0 everywhere
                    # except for the peaks that are mapped to genes that are
                    # unmasked, in which case they are 1
                    atac_decoder_addon_mask = torch.mm(
                        getattr(self,
                                f"{entity}_rna_decoder_addon_mask").to(torch.int),
                        gene_peaks_mask.to(torch.int)).to(torch.bool)
                    setattr(self,
                            f"{entity}_atac_decoder_addon_mask",
                            atac_decoder_addon_mask)

                    # Determine add-on atac idx based on peaks that are
                    # mapped to unmasked genes
                    self.features_idx_dict_[f"{entity}_addon_atac_idx"] = (
                        torch.nonzero(
                        (atac_decoder_addon_mask.sum(axis=0) > 0)
                        ).squeeze().tolist())
            else:
                # still need to initialize empty mask even if atac is not a modality
                if "atac" not in self.modalities_:
                    setattr(self,
                            f"{entity}_atac_decoder_addon_mask",
                            None)
                    self.features_idx_dict_[f"{entity}_addon_atac_idx"] = None  
                for modality in self.modalities_:
                    setattr(self,
                            f"{entity}_{modality}_decoder_addon_mask",
                            None)
                    self.features_idx_dict_[f"{entity}_addon_{modality}_idx"] = None  

        
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        print("Number of vertices in STORMModel is: " + str(self.vertices.size))
        g2v = sc.GraphEncoder(self.vertices.size, self.n_prior_gp_ + n_addon_gp)
        v2g = sc.GraphDecoder()
        self.modalities, idx, x2u, u2z, u2x_targets, u2x_sources, node_label_aggregator, all_ct = {}, {}, {}, {}, {}, {}, {}, set()
        num_timepoints = 0
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0:
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                h_depth=h_depth, h_dim=h_dim, dropout=dropout
            )
            u2z[k] = sc.SimpleDataEncoder(
                latent_dim, self.n_prior_gp_ + n_addon_gp
            )
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            data_config["timepoints"] = pd.Index([]) if data_config["timepoints"] is None \
                else pd.Index(data_config["timepoints"])
            
            if k == "rna":
                # Initialize node-label aggregator module
                node_label_aggregator[k] = sc.OneHopGCNNormNodeLabelAggregator(
                    modality="rna")
                # Initialize masked gene expression decoders
                u2x_targets[k] = sc.MaskedOmicsFeatureDecoder(
                    modality=k,
                    entity="target",
                    n_prior_gp_input=self.n_prior_gp_,
                    n_addon_gp_input=n_addon_gp,
                    n_output=self.n_output_genes_,
                    mask=self.gp_targets_mask_,
                    addon_mask=self.target_rna_decoder_addon_mask,
                    masked_features_idx=self.features_idx_dict_["target_masked_rna_idx"],
                    recon_loss="nb",
                    n_batches=max(data_config["batches"].size, 1))
                u2x_sources[k] = sc.MaskedOmicsFeatureDecoder(
                    modality=k,
                    entity="source",
                    n_prior_gp_input=self.n_prior_gp_,
                    n_addon_gp_input=n_addon_gp,
                    n_output=self.n_output_genes_,
                    mask=self.gp_sources_mask_,
                    addon_mask=self.source_rna_decoder_addon_mask,
                    masked_features_idx=self.features_idx_dict_["source_masked_rna_idx"],
                    recon_loss="nb",
                    n_batches=max(data_config["batches"].size, 1))
            elif k == "atac":
                # Initialize node-label aggregator module
                node_label_aggregator[k] = sc.OneHopGCNNormNodeLabelAggregator(
                    modality="atac")
                # Initialize masked atac decoders
                u2x_targets[k] = sc.MaskedOmicsFeatureDecoder(
                    modality=k,
                    entity="target",
                    n_prior_gp_input=self.n_prior_gp_,
                    n_addon_gp_input=n_addon_gp,
                    n_output=self.n_output_peaks_,
                    mask=self.ca_targets_mask_,
                    addon_mask=self.target_atac_decoder_addon_mask,
                    masked_features_idx=self.features_idx_dict_[
                        "target_masked_atac_idx"],
                    recon_loss="nb",
                    n_batches=max(data_config["batches"].size, 1))
                u2x_sources[k] = sc.MaskedOmicsFeatureDecoder(
                    modality=k,
                    entity="source",
                    n_prior_gp_input=self.n_prior_gp_,
                    n_addon_gp_input=n_addon_gp,
                    n_output=self.n_output_peaks_,
                    mask=self.ca_sources_mask_,
                    addon_mask=self.source_atac_decoder_addon_mask,
                    masked_features_idx=self.features_idx_dict_[
                        "source_masked_atac_idx"],
                    recon_loss="nb",
                    n_batches=max(data_config["batches"].size, 1))

            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
            num_timepoints = len(np.unique(adata.obs[data_config["use_timepoint"]]))

        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct

        # TODO: originally for input to discirminator, can remove if truly dont need
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0

        # print("du_n_batches: " + str(du_n_batches))

        du = sc.CombinedDiscriminator(
            in_features=latent_dim, n_modalities=len(self.modalities), n_timepoints=num_timepoints,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        
        prior = sc.Prior()
        super().__init__(
            g2v, v2g, x2u, u2z, u2x_targets, u2x_sources, graph_decoder, idx, du, prior, node_label_aggregator,
            u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size),
            n_input=self.n_input_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_prior_gp=self.n_prior_gp_,
            n_addon_gp=self.n_addon_gp_,
            n_output_genes=self.n_output_genes_,
            n_output_peaks=self.n_output_peaks_,
            target_rna_decoder_mask=self.gp_targets_mask_,
            source_rna_decoder_mask=self.gp_sources_mask_,
            target_atac_decoder_mask=self.ca_targets_mask_,
            source_atac_decoder_mask=self.ca_sources_mask_,
            target_rna_decoder_addon_mask=self.target_rna_decoder_addon_mask,
            source_rna_decoder_addon_mask=self.source_rna_decoder_addon_mask,
            target_atac_decoder_addon_mask=self.target_atac_decoder_addon_mask,
            source_atac_decoder_addon_mask=self.source_atac_decoder_addon_mask,
            features_idx_dict=self.features_idx_dict_,
            features_scale_factors=self.features_scale_factors_,
            gene_peaks_mask=gene_peaks_mask,
            active_gp_thresh_ratio=active_gp_thresh_ratio,
            active_gp_type=self.active_gp_type_,
            include_edge_recon_loss = self.include_edge_recon_loss_,
            include_edge_kl_loss = self.include_edge_kl_loss_
        )

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "STORMModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(  # pylint: disable=arguments-differ
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 1000.,
            lam_sup: float = 0.02,
            lam_cos: float = 0.02,
            lam_masked_l1: float = 0.02,
            lam_addon_l1: float = 0.02,
            lam_adj: float = 0.02,
            lam_tmp: float = 0.1,
            n_epochs_rna: int = 50,
            n_epochs_atac: int = 100,
            n_epochs_all_gp: int = 25,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        self.lam_masked_l1 = lam_masked_l1
        self.lam_addon_l1 = lam_addon_l1

        # automatically don't include any atac epochs if only 1 modality
        if "atac" not in self.net.keys:
            n_epochs_atac = 0

        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_cos=lam_cos, lam_masked_l1=lam_masked_l1, lam_addon_l1=lam_addon_l1,
            lam_adj=lam_adj,
            lam_tmp=lam_tmp,
            n_epochs_rna = n_epochs_rna,
            n_epochs_atac = n_epochs_atac,
            n_epochs_all_gp = n_epochs_all_gp,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None, 
            l1_targets_categories: Optional[list]=["target_gene"],
            l1_sources_categories: Optional[list]=None
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        l1_targets_categories
            Gene program mask targets categories for which l1 regularization loss
            will be applied
        l1_sources_categories
            Gene program mask sources categories for which l1 regularization loss
            will be applied
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn"
        )
        graph = GraphDataset(
            graph, self.vertices, neg_samples=neg_samples,
            weighted_sampling=True, deemphasize_loops=True
        )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        if self.lam_masked_l1 > 0.:
            # Create mask for l1 regularization loss
            targets_categories_label_encoder_key = self.anndata_keys_[
                "targets_categories_label_encoder"]
            sources_categories_label_encoder_key = self.anndata_keys_[
                "sources_categories_label_encoder"]
            if l1_targets_categories is None:
                l1_targets_categories_encoded = list(adatas["rna"].uns[
                    targets_categories_label_encoder_key].values())
            else:
                l1_targets_categories_encoded = [
                    adatas["rna"].uns[
                        targets_categories_label_encoder_key][category]
                    for category in l1_targets_categories if category in
                    adatas["rna"].uns[targets_categories_label_encoder_key]]
            if l1_sources_categories is None:
                l1_sources_categories_encoded = list(adatas["rna"].uns[
                    sources_categories_label_encoder_key].values())
            else:
                l1_sources_categories_encoded = [
                    adatas["rna"].uns[
                        sources_categories_label_encoder_key][category]
                    for category in l1_sources_categories if category in
                    adatas["rna"].uns[sources_categories_label_encoder_key]]

            gp_targets_categories_mask_key = self.anndata_keys_[
                "gp_targets_categories_mask"]
            l1_targets_mask = torch.from_numpy(np.isin(
                adatas["rna"].varm[gp_targets_categories_mask_key],
                l1_targets_categories_encoded))
            gp_sources_categories_mask_key = self.anndata_keys_[
                "gp_sources_categories_mask"]
            l1_sources_mask = torch.from_numpy(np.isin(
                adatas["rna"].varm[gp_sources_categories_mask_key],
                l1_sources_categories_encoded))
        else:
            l1_targets_mask = None
            l1_sources_mask = None

        self.l1_targets_mask = l1_targets_mask
        self.l1_sources_mask = l1_sources_mask

        super().fit(
            data, graph, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch

        Returns
        -------
        losses
            Loss function values
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        graph = GraphDataset(
            graph, self.vertices,
            neg_samples=neg_samples,
            weighted_sampling=True,
            deemphasize_loops=True
        )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        return super().get_losses(
            data, graph, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_graph(
            self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        graph_embedding
            Graph (feature) embedding
            with shape :math:`n_{feature} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{feature} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        graph = GraphDataset(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        v = self.net.g2v(eidx, enorm, esgn)
        if n_sample:
            return torch.cat([
                v.sample((1, )).cpu() for _ in range(n_sample)
            ]).permute(1, 0, 2).numpy()
        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for items, *_ in data_loader:
            x, xrep, *_ = items
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu())
            
        return torch.cat(result).numpy()

    @torch.no_grad()
    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData, graph: nx.Graph,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Decode data

        Parameters
        ----------
        source_key
            Source modality key
        target_key
            Target modality key
        adata
            Source modality data
        graph
            Guidance graph
        target_libsize
            Target modality library size, by default 1.0
        target_batch
            Target modality batch, by default batch 0
        batch_size
            Size of minibatches

        Returns
        -------
        decoded
            Decoded data

        Note
        ----
        This is EXPERIMENTAL!
        """
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, batch_size=batch_size)
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")
        if not hasattr(self, "anndata_keys_"):
            self.anndata_keys_ = dict(STORM_DEFAULT_KEYS)

    def __repr__(self) -> str:
        return (
            f"STORM model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )
    def get_active_gps(self, adata: AnnData, atac: Optional[AnnData] = None) -> np.ndarray:
        """
        Get active gene programs based on the gene expression decoder gene
        weights of gene programs. Active gene programs are gene programs
        whose absolute gene weights aggregated over all genes are greater than
        ´self.active_gp_thresh_ratio_´ times the absolute gene weights
        aggregation of the gene program with the maximum value across all gene 
        programs.

        Parameters
        ----------
        adata:
            AnnData object to get the active gene programs for. If ´None´, uses
            the adata object stored in the model instance.

        Returns
        ----------
        active_gps:
            Gene program names of active gene programs (dim: n_active_gps,)
        """
        device = next(self.net.parameters()).device

        if atac:
            active_gp_mask = self.net.get_active_gp_mask(which_modality = "atac")
        else:
            active_gp_mask = self.net.get_active_gp_mask()
        active_gp_mask = active_gp_mask.detach().cpu().numpy()

        gp_names_key = self.anndata_keys_["gp_names"]
        if self.n_addon_gp_ > 0:
            # Add add-on gps to adata
            gp_list = list(adata.uns[gp_names_key])
            for i in range(self.n_addon_gp_):
                if f"Add-on_{i}_GP" not in gp_list:
                    gp_list.append(f"Add-on_{i}_GP")
            adata.uns[gp_names_key] = np.array(gp_list)
        else:
            # Remove add-on gps from adata
            for gp_name in list(adata.uns[gp_names_key]):
                if "Add-on" in gp_name:
                    adata.uns[gp_names_key] = np.delete(
                        adata.uns[gp_names_key],
                        list(adata.uns[gp_names_key]).index(gp_name))
        active_gps = adata.uns[gp_names_key][active_gp_mask]
        return active_gps
    
    def get_gp_data(self,
                    adata: AnnData,
                    selected_gps: Optional[Union[str, list]]=None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the index of selected gene programs as well as their omics decoder
        weights.

        Parameters:
        ----------
        selected_gps:
            Names of the selected gene programs for which data should be
            retrieved.

        Returns:
        ----------
        selected_gps_idx:
            Index of the selected gene programs (dim: n_selected_gps,)
        selected_gps_rna_decoder_weights:
            Gene weights of the rna decoders of the selected gene programs
            (dim: (2 * n_genes) x n_selected_gps).
        selected_gps_atac_decoder_weights:
            Peak weights of the atac decoders of the selected gene programs
            (dim: (2 * n_peaks) x n_selected_gps).
        """
        gp_names_key = self.anndata_keys_["gp_names"]
        # Get selected gps and their index
        all_gps = list(adata.uns[gp_names_key])
        if selected_gps is None:
            selected_gps = all_gps
        elif isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps_idx = np.array([all_gps.index(gp) for gp in selected_gps])

        # Get weights of selected gps
        all_gps_rna_decoder_weights = self.net.get_gp_weights()[0]
        selected_gps_rna_decoder_weights = (
            all_gps_rna_decoder_weights[:, selected_gps_idx]
            .cpu().detach().numpy())
        
        if "atac" in self.modalities_:
            all_gps_atac_decoder_weights = self.net.get_gp_weights()[1]
            selected_gps_atac_decoder_weights = (
                all_gps_atac_decoder_weights[:, selected_gps_idx]
                .cpu().detach().numpy())
        else:
            selected_gps_atac_decoder_weights = None

        return (selected_gps_idx,
                selected_gps_rna_decoder_weights,
                selected_gps_atac_decoder_weights)

    def get_gp_summary(self, adata: AnnData, adata_atac: AnnData) -> pd.DataFrame:
        """
        Get summary information of gene programs and return it as a DataFrame.
        
        Returns
        ----------
        gp_summary_df:
            DataFrame with gene program summary information.
        """
        device = next(self.net.parameters()).device
        
        # Get source and target omics decoder weights
        _, gp_gene_weights, gp_peak_weights = self.get_gp_data(adata)

        # Normalize gp weights to get gene importances
        gp_gene_importances = np.where(
            np.abs(gp_gene_weights).sum(0) != 0,
            np.abs(gp_gene_weights) / np.abs(gp_gene_weights).sum(0),
            0)      

        # Split gene weights and importances into source and target part
        gp_gene_weights = np.transpose(gp_gene_weights)
        gp_gene_importances = np.transpose(gp_gene_importances)
        gp_gene_weights_source = gp_gene_weights[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_weights_target = gp_gene_weights[
            :, :(gp_gene_weights.shape[1] // 2)]
        gp_gene_importances_source = gp_gene_importances[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_importances_target = gp_gene_importances[
            :, :(gp_gene_weights.shape[1] // 2)]
        
        # Get source and target gene masks
        gp_gene_mask_source = np.transpose(
            np.array(self.net.source_rna_decoder_mask).T != 0)
        gp_gene_mask_target = np.transpose(
            np.array(self.net.target_rna_decoder_mask).T != 0)
        
        # Add entries to gp mask for addon gps
        if self.n_addon_gp_ > 0:
            gp_gene_addon_mask_source = np.transpose(
            self.net.source_rna_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
            gp_gene_addon_mask_target = np.transpose(
            self.net.target_rna_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
            gp_gene_mask_source = np.concatenate(
                (gp_gene_mask_source, gp_gene_addon_mask_source), axis=0)
            gp_gene_mask_target = np.concatenate(
                (gp_gene_mask_target, gp_gene_addon_mask_target), axis=0)

        # Get active gp mask
        gp_active_status = (self.net.get_active_gp_mask().cpu().detach()
                            .numpy().tolist())

        active_gps = list(self.get_active_gps(adata=adata))

        gp_names_key = self.anndata_keys_["gp_names"]
        all_gps = list(adata.uns[gp_names_key])

        # Collect info for each gp in lists of lists
        gp_names = []
        active_gp_idx = [] # Index among active gene programs
        all_gp_idx = [] # Index among all gene programs
        n_source_genes = []
        n_non_zero_source_genes = []
        n_target_genes = []
        n_non_zero_target_genes = []
        gp_source_genes = []
        gp_target_genes = []
        gp_source_genes_weights = []
        gp_target_genes_weights = []
        gp_source_genes_importances = []
        gp_target_genes_importances = []
        for (name,
             gene_mask_source,
             gene_mask_target,
             gene_weights_source,
             gene_weights_target,
             gene_importances_source,
             gene_importances_target) in zip(
                all_gps,
                gp_gene_mask_source,
                gp_gene_mask_target,
                gp_gene_weights_source,
                gp_gene_weights_target,
                gp_gene_importances_source,
                gp_gene_importances_target):
            gp_names.append(name)
            active_gp_idx.append(active_gps.index(name)
                                 if name in active_gps else np.nan)
            all_gp_idx.append(all_gps.index(name))

            # Sort source genes according to absolute weights
            gene_weights_source_sorted = []
            gene_importances_source_sorted = []
            genes_source_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_source[gene_mask_source],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_source[gene_mask_source],
                          decimals=4),
                np.around(gene_importances_source[gene_mask_source],
                          decimals=4),        
                adata.var_names[gene_mask_source].tolist()), reverse=True):
                    genes_source_sorted.append(genes)
                    gene_weights_source_sorted.append(weights)
                    gene_importances_source_sorted.append(importances)
            
            # Sort target genes according to absolute weights
            geme_weights_target_sorted = []
            gene_importances_target_sorted = []
            genes_target_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_target[gene_mask_target],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_target[gene_mask_target],
                          decimals=4),                 
                np.around(gene_importances_target[gene_mask_target],
                          decimals=4),
                adata.var_names[gene_mask_target].tolist()), reverse=True):
                    genes_target_sorted.append(genes)
                    geme_weights_target_sorted.append(weights)
                    gene_importances_target_sorted.append(importances)                 
                
            n_source_genes.append(len(genes_source_sorted))
            n_non_zero_source_genes.append(len(np.array(
                gene_weights_source_sorted).nonzero()[0]))
            n_target_genes.append(len(genes_target_sorted))
            n_non_zero_target_genes.append(len(np.array(
                geme_weights_target_sorted).nonzero()[0]))
            gp_source_genes.append(genes_source_sorted)
            gp_target_genes.append(genes_target_sorted)
            gp_source_genes_weights.append(gene_weights_source_sorted)
            gp_target_genes_weights.append(geme_weights_target_sorted)
            gp_source_genes_importances.append(gene_importances_source_sorted)
            gp_target_genes_importances.append(gene_importances_target_sorted)
   
        gp_summary_df = pd.DataFrame(
            {"gp_name": gp_names,
             "all_gp_idx": all_gp_idx,
             "gp_active": gp_active_status,
             "active_gp_idx": active_gp_idx,
             "n_source_genes": n_source_genes,
             "n_non_zero_source_genes": n_non_zero_source_genes,
             "n_target_genes": n_target_genes,
             "n_non_zero_target_genes": n_non_zero_target_genes,
             "gp_source_genes": gp_source_genes,
             "gp_target_genes": gp_target_genes,
             "gp_source_genes_weights": gp_source_genes_weights,
             "gp_target_genes_weights": gp_target_genes_weights,
             "gp_source_genes_importances": gp_source_genes_importances,
             "gp_target_genes_importances": gp_target_genes_importances})
        
        gp_summary_df["active_gp_idx"] = (
            gp_summary_df["active_gp_idx"].astype("Int64"))
        
        if "atac" in self.modalities_:
            # Add peak info for each gp
            
            # Normalize gp weights to get gene importances
            gp_peak_importances = np.where(
                np.abs(gp_peak_weights).sum(0) != 0,
                np.abs(gp_peak_weights) / np.abs(gp_peak_weights).sum(0),
                0)
        
            # Split peak weights and importances into source and target part
            gp_peak_weights = np.transpose(gp_peak_weights)
            gp_peak_importances = np.transpose(gp_peak_importances)
            gp_peak_weights_source = gp_peak_weights[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_weights_target = gp_peak_weights[
                :, :(gp_peak_weights.shape[1] // 2)]
            gp_peak_importances_source = gp_peak_importances[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_importances_target = gp_peak_importances[
                :, :(gp_peak_weights.shape[1] // 2)]

            # Get source and target peak masks
            gp_peak_mask_source = np.transpose(
                np.array(
                    self.net.source_atac_decoder_mask.to_dense()).T != 0)
            gp_peak_mask_target = np.transpose(
                np.array(
                    self.net.target_atac_decoder_mask.to_dense()).T != 0)

            # Add entries to gp mask for addon gps
            if self.n_addon_gp_ > 0:
                gp_peak_addon_mask_source = np.transpose(
                self.net.source_atac_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
                gp_peak_addon_mask_target = np.transpose(
                self.net.target_atac_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
                gp_peak_mask_source = np.concatenate(
                    (gp_peak_mask_source, gp_peak_addon_mask_source), axis=0)
                gp_peak_mask_target = np.concatenate(
                    (gp_peak_mask_target, gp_peak_addon_mask_target), axis=0)

            # Collect info for each gp in lists of lists
            n_source_peaks = []
            n_non_zero_source_peaks = []
            n_target_peaks = []
            n_non_zero_target_peaks = []
            gp_source_peaks = []
            gp_target_peaks = []
            gp_source_peaks_weights = []
            gp_target_peaks_weights = []
            gp_source_peaks_importances = []
            gp_target_peaks_importances = []
            for (gp_source_peaks_idx,
                 gp_target_peaks_idx,
                 gp_source_peaks_weights_arr,
                 gp_target_peaks_weights_arr,
                 gp_source_peaks_importances_arr,
                 gp_target_peaks_importances_arr) in zip(
                    gp_peak_mask_source,
                    gp_peak_mask_target,
                    gp_peak_weights_source,
                    gp_peak_weights_target,
                    gp_peak_importances_source,
                    gp_peak_importances_target):
                # Sort source peaks according to absolute weights
                peak_weights_source_sorted = []
                peak_importances_source_sorted = []
                peaks_source_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                                    decimals=4)), # just for sorting
                    np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                            decimals=4),
                    np.around(gp_source_peaks_importances_arr[gp_source_peaks_idx],
                            decimals=4),        
                    adata_atac.var_names[gp_source_peaks_idx].tolist()),reverse=True):
                        peaks_source_sorted.append(peaks)
                        peak_weights_source_sorted.append(weights)
                        peak_importances_source_sorted.append(importances)
                
                # Sort target peaks according to absolute weights
                peak_weights_target_sorted = []
                peak_importances_target_sorted = []
                peaks_target_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                                    decimals=4)),
                    np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                            decimals=4),                 
                    np.around(gp_target_peaks_importances_arr[gp_target_peaks_idx],
                            decimals=4),
                    adata_atac.var_names[gp_target_peaks_idx].tolist()), reverse=True):
                        peaks_target_sorted.append(peaks)
                        peak_weights_target_sorted.append(weights)
                        peak_importances_target_sorted.append(importances)                 
                    
                n_source_peaks.append(len(peaks_source_sorted))
                n_non_zero_source_peaks.append(len(np.array(
                    peak_weights_source_sorted).nonzero()[0]))
                n_target_peaks.append(len(peaks_target_sorted))
                n_non_zero_target_peaks.append(len(np.array(
                    peak_weights_target_sorted).nonzero()[0]))
                gp_source_peaks.append(peaks_source_sorted)
                gp_target_peaks.append(peaks_target_sorted)
                gp_source_peaks_weights.append(peak_weights_source_sorted)
                gp_target_peaks_weights.append(peak_weights_target_sorted)
                gp_source_peaks_importances.append(peak_importances_source_sorted)
                gp_target_peaks_importances.append(peak_importances_target_sorted)

            gp_summary_df["n_source_peaks"] = n_source_peaks
            gp_summary_df["n_non_zero_source_peaks"] = n_non_zero_source_peaks
            gp_summary_df["n_target_peaks"] = n_target_peaks
            gp_summary_df["n_non_zero_target_peaks"] = n_non_zero_target_peaks
            gp_summary_df["gp_source_peaks"] = gp_source_peaks
            gp_summary_df["gp_target_peaks"] = gp_target_peaks
            gp_summary_df["gp_source_peaks_weights"] = gp_source_peaks_weights
            gp_summary_df["gp_target_peaks_weights"] = gp_target_peaks_weights
            gp_summary_df["gp_source_peaks_importances"] = gp_source_peaks_importances
            gp_summary_df["gp_target_peaks_importances"] = gp_target_peaks_importances
            gp_summary_df["gp_source_peaks_importances"] = (
                gp_summary_df["gp_source_peaks_importances"].replace(np.nan, 0.))
            gp_summary_df["gp_target_peaks_importances"] = (
                gp_summary_df["gp_target_peaks_importances"].replace(np.nan, 0.))

        return gp_summary_df

    
    def get_gp_summary_atac(self, adata: AnnData, adata_atac: AnnData) -> pd.DataFrame:
        """
        Get summary information of gene programs and return it as a DataFrame.
        
        Returns
        ----------
        gp_summary_df:
            DataFrame with gene program summary information.
        """
        device = next(self.net.parameters()).device
        
        # Get source and target omics decoder weights
        _, gp_gene_weights, gp_peak_weights = self.get_gp_data(adata)

        # Normalize gp weights to get gene importances
        gp_gene_importances = np.where(
            np.abs(gp_gene_weights).sum(0) != 0,
            np.abs(gp_gene_weights) / np.abs(gp_gene_weights).sum(0),
            0)      

        # Split gene weights and importances into source and target part
        gp_gene_weights = np.transpose(gp_gene_weights)
        gp_gene_importances = np.transpose(gp_gene_importances)
        gp_gene_weights_source = gp_gene_weights[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_weights_target = gp_gene_weights[
            :, :(gp_gene_weights.shape[1] // 2)]
        gp_gene_importances_source = gp_gene_importances[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_importances_target = gp_gene_importances[
            :, :(gp_gene_weights.shape[1] // 2)]
        
        # Get source and target gene masks
        gp_gene_mask_source = np.transpose(
            np.array(self.net.source_rna_decoder_mask).T != 0)
        gp_gene_mask_target = np.transpose(
            np.array(self.net.target_rna_decoder_mask).T != 0)
        
        # Add entries to gp mask for addon gps
        if self.n_addon_gp_ > 0:
            gp_gene_addon_mask_source = np.transpose(
            self.net.source_rna_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
            gp_gene_addon_mask_target = np.transpose(
            self.net.target_rna_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
            gp_gene_mask_source = np.concatenate(
                (gp_gene_mask_source, gp_gene_addon_mask_source), axis=0)
            gp_gene_mask_target = np.concatenate(
                (gp_gene_mask_target, gp_gene_addon_mask_target), axis=0)

        # Get active gp mask
        gp_active_status = (self.net.get_active_gp_mask(which_modality = "atac").cpu().detach()
                            .numpy().tolist())

        active_gps = list(self.get_active_gps(adata, adata_atac))

        gp_names_key = self.anndata_keys_["gp_names"]
        all_gps = list(adata.uns[gp_names_key])

        # Collect info for each gp in lists of lists
        gp_names = []
        active_gp_idx = [] # Index among active gene programs
        all_gp_idx = [] # Index among all gene programs
        n_source_genes = []
        n_non_zero_source_genes = []
        n_target_genes = []
        n_non_zero_target_genes = []
        gp_source_genes = []
        gp_target_genes = []
        gp_source_genes_weights = []
        gp_target_genes_weights = []
        gp_source_genes_importances = []
        gp_target_genes_importances = []
        for (name,
             gene_mask_source,
             gene_mask_target,
             gene_weights_source,
             gene_weights_target,
             gene_importances_source,
             gene_importances_target) in zip(
                all_gps,
                gp_gene_mask_source,
                gp_gene_mask_target,
                gp_gene_weights_source,
                gp_gene_weights_target,
                gp_gene_importances_source,
                gp_gene_importances_target):
            gp_names.append(name)
            active_gp_idx.append(active_gps.index(name)
                                 if name in active_gps else np.nan)
            all_gp_idx.append(all_gps.index(name))

            # Sort source genes according to absolute weights
            gene_weights_source_sorted = []
            gene_importances_source_sorted = []
            genes_source_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_source[gene_mask_source],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_source[gene_mask_source],
                          decimals=4),
                np.around(gene_importances_source[gene_mask_source],
                          decimals=4),        
                adata.var_names[gene_mask_source].tolist()), reverse=True):
                    genes_source_sorted.append(genes)
                    gene_weights_source_sorted.append(weights)
                    gene_importances_source_sorted.append(importances)
            
            # Sort target genes according to absolute weights
            geme_weights_target_sorted = []
            gene_importances_target_sorted = []
            genes_target_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_target[gene_mask_target],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_target[gene_mask_target],
                          decimals=4),                 
                np.around(gene_importances_target[gene_mask_target],
                          decimals=4),
                adata.var_names[gene_mask_target].tolist()), reverse=True):
                    genes_target_sorted.append(genes)
                    geme_weights_target_sorted.append(weights)
                    gene_importances_target_sorted.append(importances)                 
                
            n_source_genes.append(len(genes_source_sorted))
            n_non_zero_source_genes.append(len(np.array(
                gene_weights_source_sorted).nonzero()[0]))
            n_target_genes.append(len(genes_target_sorted))
            n_non_zero_target_genes.append(len(np.array(
                geme_weights_target_sorted).nonzero()[0]))
            gp_source_genes.append(genes_source_sorted)
            gp_target_genes.append(genes_target_sorted)
            gp_source_genes_weights.append(gene_weights_source_sorted)
            gp_target_genes_weights.append(geme_weights_target_sorted)
            gp_source_genes_importances.append(gene_importances_source_sorted)
            gp_target_genes_importances.append(gene_importances_target_sorted)
   
        gp_summary_df = pd.DataFrame(
            {"gp_name": gp_names,
             "all_gp_idx": all_gp_idx,
             "gp_active": gp_active_status,
             "active_gp_idx": active_gp_idx,
             "n_source_genes": n_source_genes,
             "n_non_zero_source_genes": n_non_zero_source_genes,
             "n_target_genes": n_target_genes,
             "n_non_zero_target_genes": n_non_zero_target_genes,
             "gp_source_genes": gp_source_genes,
             "gp_target_genes": gp_target_genes,
             "gp_source_genes_weights": gp_source_genes_weights,
             "gp_target_genes_weights": gp_target_genes_weights,
             "gp_source_genes_importances": gp_source_genes_importances,
             "gp_target_genes_importances": gp_target_genes_importances})
        
        gp_summary_df["active_gp_idx"] = (
            gp_summary_df["active_gp_idx"].astype("Int64"))
        
        if "atac" in self.modalities_:
            # Add peak info for each gp
            
            # Normalize gp weights to get gene importances
            gp_peak_importances = np.where(
                np.abs(gp_peak_weights).sum(0) != 0,
                np.abs(gp_peak_weights) / np.abs(gp_peak_weights).sum(0),
                0)
        
            # Split peak weights and importances into source and target part
            gp_peak_weights = np.transpose(gp_peak_weights)
            gp_peak_importances = np.transpose(gp_peak_importances)
            gp_peak_weights_source = gp_peak_weights[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_weights_target = gp_peak_weights[
                :, :(gp_peak_weights.shape[1] // 2)]
            gp_peak_importances_source = gp_peak_importances[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_importances_target = gp_peak_importances[
                :, :(gp_peak_weights.shape[1] // 2)]

            # Get source and target peak masks
            gp_peak_mask_source = np.transpose(
                np.array(
                    self.net.source_atac_decoder_mask.to_dense()).T != 0)
            gp_peak_mask_target = np.transpose(
                np.array(
                    self.net.target_atac_decoder_mask.to_dense()).T != 0)

            # Add entries to gp mask for addon gps
            if self.n_addon_gp_ > 0:
                gp_peak_addon_mask_source = np.transpose(
                self.net.source_atac_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
                gp_peak_addon_mask_target = np.transpose(
                self.net.target_atac_decoder_addon_mask[0].cpu().detach().numpy().T != 0)
                gp_peak_mask_source = np.concatenate(
                    (gp_peak_mask_source, gp_peak_addon_mask_source), axis=0)
                gp_peak_mask_target = np.concatenate(
                    (gp_peak_mask_target, gp_peak_addon_mask_target), axis=0)

            # Collect info for each gp in lists of lists
            n_source_peaks = []
            n_non_zero_source_peaks = []
            n_target_peaks = []
            n_non_zero_target_peaks = []
            gp_source_peaks = []
            gp_target_peaks = []
            gp_source_peaks_weights = []
            gp_target_peaks_weights = []
            gp_source_peaks_importances = []
            gp_target_peaks_importances = []
            for (gp_source_peaks_idx,
                 gp_target_peaks_idx,
                 gp_source_peaks_weights_arr,
                 gp_target_peaks_weights_arr,
                 gp_source_peaks_importances_arr,
                 gp_target_peaks_importances_arr) in zip(
                    gp_peak_mask_source,
                    gp_peak_mask_target,
                    gp_peak_weights_source,
                    gp_peak_weights_target,
                    gp_peak_importances_source,
                    gp_peak_importances_target):
                # Sort source peaks according to absolute weights
                peak_weights_source_sorted = []
                peak_importances_source_sorted = []
                peaks_source_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                                    decimals=4)), # just for sorting
                    np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                            decimals=4),
                    np.around(gp_source_peaks_importances_arr[gp_source_peaks_idx],
                            decimals=4),        
                    adata_atac.var_names[gp_source_peaks_idx].tolist()),reverse=True):
                        peaks_source_sorted.append(peaks)
                        peak_weights_source_sorted.append(weights)
                        peak_importances_source_sorted.append(importances)
                
                # Sort target peaks according to absolute weights
                peak_weights_target_sorted = []
                peak_importances_target_sorted = []
                peaks_target_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                                    decimals=4)),
                    np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                            decimals=4),                 
                    np.around(gp_target_peaks_importances_arr[gp_target_peaks_idx],
                            decimals=4),
                    adata_atac.var_names[gp_target_peaks_idx].tolist()), reverse=True):
                        peaks_target_sorted.append(peaks)
                        peak_weights_target_sorted.append(weights)
                        peak_importances_target_sorted.append(importances)                 
                    
                n_source_peaks.append(len(peaks_source_sorted))
                n_non_zero_source_peaks.append(len(np.array(
                    peak_weights_source_sorted).nonzero()[0]))
                n_target_peaks.append(len(peaks_target_sorted))
                n_non_zero_target_peaks.append(len(np.array(
                    peak_weights_target_sorted).nonzero()[0]))
                gp_source_peaks.append(peaks_source_sorted)
                gp_target_peaks.append(peaks_target_sorted)
                gp_source_peaks_weights.append(peak_weights_source_sorted)
                gp_target_peaks_weights.append(peak_weights_target_sorted)
                gp_source_peaks_importances.append(peak_importances_source_sorted)
                gp_target_peaks_importances.append(peak_importances_target_sorted)

            gp_summary_df["n_source_peaks"] = n_source_peaks
            gp_summary_df["n_non_zero_source_peaks"] = n_non_zero_source_peaks
            gp_summary_df["n_target_peaks"] = n_target_peaks
            gp_summary_df["n_non_zero_target_peaks"] = n_non_zero_target_peaks
            gp_summary_df["gp_source_peaks"] = gp_source_peaks
            gp_summary_df["gp_target_peaks"] = gp_target_peaks
            gp_summary_df["gp_source_peaks_weights"] = gp_source_peaks_weights
            gp_summary_df["gp_target_peaks_weights"] = gp_target_peaks_weights
            gp_summary_df["gp_source_peaks_importances"] = gp_source_peaks_importances
            gp_summary_df["gp_target_peaks_importances"] = gp_target_peaks_importances
            gp_summary_df["gp_source_peaks_importances"] = (
                gp_summary_df["gp_source_peaks_importances"].replace(np.nan, 0.))
            gp_summary_df["gp_target_peaks_importances"] = (
                gp_summary_df["gp_target_peaks_importances"].replace(np.nan, 0.))

        return gp_summary_df
    
    def encode_gp_latent(
            self, key: str, adata: AnnData, graph: nx.Graph, batch_size: int = 128,
            n_sample: Optional[int] = None, only_active_gps: bool=True, sign_adjusted: bool=False, adata_rna: Optional[AnnData] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        z_encoder = self.net.u2z[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for items, *_ in data_loader:
            x, xrep, *_ = items
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            if n_sample:
                temp_result = u.sample((n_sample, )).permute(1, 0, 2)
            else:
                temp_result = u.mean

            z = z_encoder(temp_result)
            z = z.detach().cpu()

            active_gp_mask = None
            if only_active_gps:
                # Filter to active gene programs only
                if key == "rna":
                    active_gp_mask = self.net.get_active_gp_mask().detach().cpu()
                elif key == "atac":
                    active_gp_mask = self.net.get_active_gp_mask(which_modality = "atac").detach().cpu()
                z = z[:, active_gp_mask]

            result.append(z)
        # N x N_GP
        result = torch.cat(result)

        # TODO: include a case for when sign_adjusted is False
        if sign_adjusted:
            if key == "rna":
                # call get_active_gp to adjust number of GPs (adding the denovo GP names in .uns)
                _ = self.get_active_gps(adata) 
                # N_genes x N_GP
                _, gp_gene_weights, gp_peak_weights = self.get_gp_data(adata)
                
                # TODO: deal with source
                # Split gene weights and importances into source and target part
                gp_gene_weights = np.transpose(gp_gene_weights)
                # N_GP x N_genes
                gp_gene_weights_source = gp_gene_weights[
                    :, (gp_gene_weights.shape[1] // 2):]
                gp_gene_weights_target = gp_gene_weights[
                    :, :(gp_gene_weights.shape[1] // 2)]

                # signs = np.sign(gp_gene_weights_target)

                # # Mask to filter weights with absolute value >= 0.001
                # threshold_mask = np.abs(gp_gene_weights_target) >= 0.001
                # filtered_weights = np.where(threshold_mask, gp_gene_weights_target, 0)
                # signs = np.sign(filtered_weights)

                abs_weights = np.abs(gp_gene_weights_target)
                # Step 1: Compute median absolute weights for each gene program (row)
                median_weights = np.median(abs_weights, axis=1)
                # Step 2: Create a boolean mask where gene weights exceed the median for each row
                mask = abs_weights > median_weights[:, np.newaxis]
                # Step 3: Get the signs of gene weights that exceed the median
                signs = np.sign(gp_gene_weights_target) * mask


            elif key == "atac":
                _ = self.get_active_gps(adata_rna, atac = adata) 
                _, gp_gene_weights, gp_peak_weights = self.get_gp_data(adata_rna)

                # TODO: deal with source
                # Split gene weights and importances into source and target part
                gp_peak_weights = np.transpose(gp_peak_weights)
                # N_GP x N_genes
                gp_peak_weights_source = gp_peak_weights[
                    :, (gp_peak_weights.shape[1] // 2):]
                gp_peak_weights_target = gp_peak_weights[
                    :, :(gp_peak_weights.shape[1] // 2)]
                
                # signs = np.sign(gp_peak_weights_target)

                # # Mask to filter weights with absolute value >= 0.001
                # threshold_mask = np.abs(gp_peak_weights_target) >= 0.001
                # filtered_weights = np.where(threshold_mask, gp_peak_weights_target, 0)
                # signs = np.sign(filtered_weights)

                abs_weights = np.abs(gp_peak_weights_target)
                # Step 1: Compute median absolute weights for each gene program (row)
                median_weights = np.median(abs_weights, axis=1)
                # Step 2: Create a boolean mask where gene weights exceed the median for each row
                mask = abs_weights > median_weights[:, np.newaxis]
                # Step 3: Get the signs of gene weights that exceed the median
                signs = np.sign(gp_peak_weights_target) * mask

            
            # If the sum is positive, the majority are positive, otherwise, the majority are negative
            majority_sign = np.sign(signs.sum(axis=1))
            # Replace 0s with 1s (assuming ties should be considered as majority positive)
            majority_sign[majority_sign == 0] = 1
            if active_gp_mask is not None:
                majority_sign = majority_sign[active_gp_mask]

            majority_sign = torch.tensor(majority_sign, dtype=torch.float32)
            result_sign_corrected = result * majority_sign

        return result.numpy(), result_sign_corrected.numpy(), majority_sign.numpy()

@logged
class PairedSTORMModel(STORMModel, nn.Module):

    r"""
    GLUE model for partially-paired single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    TRAINER_TYPE = PairedSTORMTrainer

    def compile(  # pylint: disable=arguments-renamed
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 1000,
            lam_sup: float = 0.02,
            lam_cos: float = 0.02,
            lam_masked_l1: float = 0.02,
            lam_addon_l1: float = 0.02,
            lam_adj: float = 0.02,
            lam_tmp: float = 0.1,
            n_epochs_rna: int = 50,
            n_epochs_atac: int = 100,
            n_epochs_all_gp: int = 25,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        lam_cos
            Cosine similarity weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        """
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_cos=lam_cos, lam_masked_l1 = lam_masked_l1, lam_addon_l1 = lam_addon_l1,
            lam_adj=lam_adj,
            lam_tmp = lam_tmp,
            n_epochs_rna = n_epochs_rna,
            n_epochs_atac = n_epochs_atac,
            n_epochs_all_gp = n_epochs_all_gp,
            normalize_u=normalize_u, modality_weight=modality_weight,
            lr=lr, **kwargs
        )
