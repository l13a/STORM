r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Optional, Tuple, List, Literal

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
import math

from ..num import EPS
from . import glue
from .nn import GraphConv
from .prob import ZILN, ZIN, ZINB



#-------------------------- Network modules for GLUE ---------------------------

class GraphEncoder(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(glue.GraphDecoder):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):

    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xrep
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xrep` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xrep is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xrep is used instead.
        """
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l

class SimpleDataEncoder(glue.DataEncoder):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear(u))
        return h


class VanillaDataEncoder(DataEncoder):

    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x


class NBDataEncoder(DataEncoder):

    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(glue.DataDecoder):

    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        b
            Batch index 
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    r"""
    Zero-inflated normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZIN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )

class OneHopGCNNormNodeLabelAggregator(torch.nn.Module):
    """
    One-hop GCN Norm Node Label Aggregator class that uses a symmetrically
    normalized adjacency matrix (xadj) to aggregate omics feature vectors from
    a node's 1-hop neighbors.

    Modality:
        Omics modality that is aggregated. Can be either `rna` or `atac`.
    """
    def __init__(self, modality: Literal["rna", "atac"]):
        super().__init__()
        print(f"ONE HOP GCN NORM {modality.upper()} NODE LABEL AGGREGATOR")

    def forward(self, x: torch.Tensor, xadj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the One-hop GCN Norm Node Label Aggregator.
        
        Parameters
        ----------
        x : torch.Tensor
            Omics feature matrix of shape (n_nodes_batch, n_node_features).
        xadj : torch.Tensor
            Symmetrically normalized adjacency matrix of shape (n_nodes_batch, n_nodes_batch).

        Returns
        ----------
        x_neighbors : torch.Tensor
            Tensor containing the aggregated node labels for the batch.
            Shape: (n_nodes_batch, n_node_features).
        """
        # Perform GCN aggregation by multiplying normalized adjacency matrix with node features
        x_neighbors = torch.matmul(xadj, x)
        return x_neighbors

class MaskedLinear(torch.nn.Module):
    """
    Masked linear class.
    
    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/master/scarches/models/expimap/modules.py#L9;
    01.10.2022.

    Uses static and dynamic binary masks to mask connections from the input
    layer to the output layer so that only unmasked connections can be used.

    Parameters
    ----------
    n_input:
        Number of input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    mask:
        Static mask that is used to mask the node connections from the input
        layer to the output layer.
    bias:
        If ´True´, use a bias.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 bias: bool=False,
                 n_batches: int = 1):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                            " (n_input x n_output). Please provide a mask with"
                            f"  dimensions ({n_input} x {n_output}).")
        super().__init__()

        self.register_buffer("mask", mask.t())

        # Initialize a weight tensor for each batch
        # 2 x N_genes x N_gp
        self.weights = nn.Parameter(torch.Tensor(n_batches, n_output, n_input))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
        if bias:
            # 2 x N_genes
            self.bias = nn.Parameter(torch.Tensor(n_batches, n_output))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # Zero out weights with the mask so that the optimizer does not
        # consider them
        with torch.no_grad():
            for b in range(n_batches):
                self.weights[b].data *= self.mask #self.mask shape: N_genes x N_gp

    def forward(self,
                input: torch.Tensor,
                v: torch.Tensor,
                b: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the masked linear class.

        Parameters
        ----------
        input:
            Tensor containing the input features to the masked linear class.
        dynamic_mask:
            Additional optional Tensor containing a mask that changes
            during training.

        Returns
        ----------
        output:
            Tensor containing the output of the masked linear class (linear
            transformation of the input by only considering unmasked
            connections).
        """
        # input: (B, N_in), v: (N_out, N_in), b: (B,) per-cell batch indicator.
        # The math is: output[c, j] = sum_k input[c, k] * weights[b[c], j, k]
        #                                          * mask[j, k] * v[j, k]
        # A direct broadcast (``self.weights[b] * self.mask * v``) materialises
        # a (B, N_out, N_in) tensor — easily multi-GB on realistic inputs and
        # the dominant source of CUDA OOM during backward. We instead do one
        # small GEMM per batch-effect group so the largest intermediate is
        # (N_out, N_in) regardless of B.
        n_batches = self.weights.shape[0]
        n_output = self.weights.shape[1]
        output = torch.zeros(input.shape[0], n_output,
                             device=input.device, dtype=input.dtype)
        dm_t = (dynamic_mask.t().to(self.mask.device)
                if dynamic_mask is not None else None)
        for i in range(n_batches):
            sel = (b == i)
            if not bool(sel.any()):
                continue
            if dm_t is not None:
                W_i = self.weights[i] * self.mask * dm_t * v  # (N_out, N_in)
            else:
                W_i = self.weights[i] * self.mask * v          # (N_out, N_in)
            output[sel] = input[sel] @ W_i.t()
        output = output + self.bias[b]                          # (B, N_out)
        return output
    
class AddOnMaskedLayer(torch.nn.Module):
    """
    Add-on masked layer class. 
    
    Parts of the implementation are adapted from 
    https://github.com/theislab/scarches/blob/7980a187294204b5fb5d61364bb76c0b809eb945/scarches/models/expimap/modules.py#L28;
    01.10.2022.

    Parameters
    ----------
    n_input:
        Number of mask input nodes to the add-on masked layer.
    n_output:
        Number of output nodes from the add-on masked layer.
    mask:
        Mask that is used to mask the node connections for mask inputs from the
        input layer to the output layer.
    addon_mask:
        Mask that is used to mask the node connections for add-on inputs from
        the input layer to the output layer.
    masked_features_idx:
        Index of input features that are included in the mask.
    bias:
        If ´True´, use a bias for the mask input nodes.
    n_addon_input:
        Number of add-on input nodes to the add-on masked layer.
    activation:
        Activation function used at the end of the ad-on masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 addon_mask: torch.Tensor,
                 masked_features_idx: List,
                 bias: bool=False,
                 n_addon_input: int=0,
                 activation: nn.Module=nn.Softmax(dim=-1),
                 n_batches: int = 1):
        super().__init__()
        # N_pr_gp
        self.n_input = n_input
        # N_addon
        self.n_addon_input = n_addon_input
        # 2777
        self.masked_features_idx = masked_features_idx

        # Masked layer
        # n_output = N_genes
        self.masked_l = MaskedLinear(n_input=n_input,
                                     n_output=n_output,
                                     mask=mask,
                                     bias=bias,
                                     n_batches = n_batches)

        # Add-on layer
        if n_addon_input != 0:
            self.addon_l = MaskedLinear(n_input=n_addon_input,
                                        n_output=n_output,
                                        mask=addon_mask,
                                        bias=bias,
                                        n_batches = n_batches)
        
        self.activation = activation

    def forward(self,
                input: torch.Tensor,
                v: torch.Tensor,
                b: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the add-on masked layer.

        Parameters
        ----------
        input:
            Input features to the add-on masked layer. Includes add-on input
            nodes and categorical covariates embedding input nodes if specified.
        dynamic_mask:
            Additional optional dynamic mask for the masked layer.

        Returns
        ----------
        output:
            Output of the add-on masked layer.
        """
        if self.n_addon_input == 0:
            mask_input = input
            mask_v = v
        elif self.n_addon_input != 0:
            mask_input, addon_input = torch.split(
                input,
                [self.n_input, self.n_addon_input],
                dim=1)  
            mask_v, addon_v = torch.split(
                v,
                [self.n_input, self.n_addon_input],
                dim=1)  

        output = self.masked_l(
            input=mask_input,
            v=mask_v,
            b=b,
            dynamic_mask=(dynamic_mask[:self.n_input, :] if
                          dynamic_mask is not None else None)) 
            # Dynamic mask also has entries for add-on gps
        if self.n_addon_input != 0:
            # Only unmasked features will have weights != 0.
            output += self.addon_l(
                input=addon_input,
                v=addon_v,
                b=b,
                dynamic_mask=(dynamic_mask[self.n_input:, :] if
                              dynamic_mask is not None else None))        
        output = self.activation(output)
        return output

class MaskedOmicsFeatureDecoder(torch.nn.Module):
    """
    Masked omics feature decoder class.

    Takes the latent space features z (gp scores) as input, and has a masked
    layer to decode the parameters of the underlying omics feature distributions.

    Parameters
    ----------
    modality:
        Omics modality that is decoded. Can be either `rna` or `atac`.
    entity:
        Entity that is decoded. Can be either `target` or `source`.
    n_prior_gp_input (N_pr):
        Number of maskable prior gp input nodes to the decoder (maskable latent
        space dimensionality).
    n_addon_gp_input (N_nv):
        Number of non-maskable add-on gp input nodes to the decoder (
        non-maskable latent space dimensionality).
    n_output (N_rna):
        Number of output nodes from the decoder (number of omics features).
    mask:
        Mask that determines which masked input nodes / prior gp latent features
        z can contribute to the reconstruction of which omics features.
    addon_mask:
        Mask that determines which add-on input nodes / add-on gp latent
        features z can contribute to the reconstruction of which omics features.
    masked_features_idx:
        Index of omics features that are included in the mask.
    recon_loss:
        The loss used for omics reconstruction. If `nb`, uses a negative
        binomial loss.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"],
                 entity: Literal["target", "source"],
                 n_prior_gp_input: int,
                 n_addon_gp_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 addon_mask: torch.Tensor,
                 masked_features_idx: List,
                 recon_loss: Literal["nb"],
                 n_batches: int = 1):
        super().__init__()
        print(f"MASKED {entity.upper()} {modality.upper()} DECODER -> "
              f"n_prior_gp_input: {n_prior_gp_input}, "
              f"n_addon_gp_input: {n_addon_gp_input}, "
              f"n_output: {n_output}")

        self.masked_features_idx = masked_features_idx
        self.recon_loss = recon_loss

        self.nb_means_normalized_decoder = AddOnMaskedLayer(
            n_input=n_prior_gp_input,
            n_addon_input=n_addon_gp_input,
            n_output=n_output,
            bias=True,
            mask=mask,
            addon_mask=addon_mask,
            masked_features_idx=masked_features_idx,
            activation=nn.Softmax(dim=-1),
            n_batches = n_batches)

    def forward(self, u: torch.Tensor, v: torch.Tensor,
                b: torch.Tensor, log_library_size: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:   
        nb_means_normalized = self.nb_means_normalized_decoder(
            input=u,
            v=v,
            b=b,
            dynamic_mask=dynamic_mask)
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        return nb_means

def compute_cosine_similarity(tensor1: torch.Tensor,
                              tensor2: torch.Tensor,
                              eps: float=1e-8) -> torch.Tensor:
    """
    Compute the element-wise cosine similarity between two 2D tensors.

    Parameters
    ----------
    tensor1:
        First tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).
    tensor2:
        Second tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).
    
    Returns
    ----------
    cosine_sim:
        Result tensor that contains the computed element-wise cosine
        similarities (dim: n_obs).
    """
    tensor1_norm = tensor1.norm(dim=1)[:, None]
    tensor2_norm = tensor2.norm(dim=1)[:, None]
    tensor1_normalized = tensor1 / torch.max(
            tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normalized = tensor2 / torch.max(
            tensor2_norm, eps * torch.ones_like(tensor2_norm))
    cosine_sim = torch.mul(tensor1_normalized, tensor2_normalized).sum(1)
    return cosine_sim
    
class CosineSimGraphDecoder(torch.nn.Module):
    """
    Cosine similarity graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    target nodes as input, and calculates the element-wise cosine similarity
    between source and target nodes to return the reconstructed edge logits.
    
    The sigmoid activation function to compute reconstructed edge probabilities
    is integrated into the binary cross entropy loss for computational
    efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self,
                 dropout_rate: float=0.):
        super().__init__()
        print("COSINE SIM GRAPH DECODER -> "
              f"dropout_rate: {dropout_rate}")

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine similarity graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and target nodes
            (dim: 4 * edge_batch_size x n_gps due to negative edges).

        Returns
        ----------
        edge_recon_logits:
            Reconstructed edge logits (dim: 2 * edge_batch_size due to negative
            edges).
        """
        z = self.dropout(z)

        # Compute element-wise cosine similarity
        edge_recon_logits = compute_cosine_similarity(
            z[:int(z.shape[0]/2)], # ´edge_label_index[0]´
            z[int(z.shape[0]/2):]) # ´edge_label_index[1]´
        return edge_recon_logits

class ZINBDataDecoder(NBDataDecoder):

    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZINB:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class Discriminator(torch.nn.Sequential, glue.Discriminator):

    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class CombinedDiscriminator(torch.nn.Sequential, glue.Discriminator):
    r"""
    Combined modality and timepoint discriminator.

    Parameters
    ----------
    in_features : int
        Input dimensionality of the latent space.
    n_modalities : int
        Number of modality classes.
    n_timepoints : int
        Number of timepoint classes.
    h_depth : int
        Hidden layer depth.
    h_dim : int
        Hidden layer dimensionality.
    dropout : float
        Dropout rate.
    """
    def __init__(
        self, in_features: int, n_modalities: int, n_timepoints: int,
        h_depth: int = 2, h_dim: int = 256,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        # Build shared layers using torch.nn.Sequential
        od = collections.OrderedDict()
        ptr_dim = in_features
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        self.shared_layers = torch.nn.Sequential(od)  # Store shared layers separately
        
        # 2. Modality-specific subnetwork
        self.modality_block = torch.nn.Linear(h_dim, n_modalities)

        # 3. Timepoint-specific subnetwork
        self.timepoint_block = torch.nn.Linear(h_dim, n_timepoints)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the combined discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input latent space (batch_size x latent_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Modality logits and timepoint logits.
        """
        # Forward pass through shared layers
        features = self.shared_layers(x)  # Only process shared layers here

        # Separate outputs for modality and timepoint
        modality_logits = self.modality_block(features) 
        timepoint_logits = self.timepoint_block(features) 
        
        return modality_logits, timepoint_logits

class Classifier(torch.nn.Linear):

    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


class Prior(glue.Prior):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


