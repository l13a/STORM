from typing import List, Literal, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def negative_sampling_loss(cosine_similarity_matrix, adj_matrix, num_negative_samples=16):
    """
    Compute BCE loss using positive and negative samples.
    
    Parameters:
    - cosine_similarity_matrix: The cosine similarity matrix for the batch (batch_size x batch_size).
    - adj_matrix: The ground truth adjacency matrix for the batch (batch_size x batch_size).
    - num_negative_samples: Number of negative samples to use for each batch.
    
    Returns:
    - loss: The binary cross-entropy loss with negative sampling.
    """
    
    # Get the positive sample indices where there are edges
    positive_indices = (adj_matrix == 1).nonzero(as_tuple=False)  # Indices of positive edges
    
    # Get the negative sample indices where there are no edges
    negative_indices = (adj_matrix == 0).nonzero(as_tuple=False)  # Indices of non-edges
    
    # Randomly sample negative examples
    num_neg_samples = min(num_negative_samples, len(negative_indices))
    negative_indices_sampled = negative_indices[torch.randint(0, len(negative_indices), (num_neg_samples,))]

    # Gather the cosine similarities for the positive and negative samples
    positive_sim = cosine_similarity_matrix[positive_indices[:, 0], positive_indices[:, 1]]  # Positives
    negative_sim = cosine_similarity_matrix[negative_indices_sampled[:, 0], negative_indices_sampled[:, 1]]  # Negatives

    # Apply sigmoid to the similarities to get probabilities
    positive_probs = torch.sigmoid(positive_sim)
    negative_probs = torch.sigmoid(negative_sim)

    # Create the labels: 1 for positive, 0 for negative
    positive_labels = torch.ones_like(positive_probs)
    negative_labels = torch.zeros_like(negative_probs)

    # Combine positive and negative samples
    all_probs = torch.cat([positive_probs, negative_probs])
    all_labels = torch.cat([positive_labels, negative_labels])

    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy(all_probs, all_labels)
    
    return loss

def compute_omics_recon_nb_loss(x: torch.Tensor,
                                mu: torch.Tensor,
                                theta: torch.Tensor,
                                eps: float=1e-8) -> torch.Tensor:
    """
    Compute omics reconstruction loss according to a negative binomial model,
    which is often used to model omics count data such as scRNA-seq or
    scATAC-seq data.

    Parts of the implementation are adapted from Lopez, R., Regier, J., Cole, M.
    B., Jordan, M. I. & Yosef, N. Deep generative modeling for single-cell
    transcriptomics. Nat. Methods 15, 1053–1058 (2018) ->
    https://github.com/scverse/scvi-tools/blob/main/scvi/distributions/_negative_binomial.py#L75;
    29.11.2022.

    Parameters
    ----------
    x:
        Reconstructed feature vector (dim: batch_size, n_genes; nodes that
        are in current batch beyond originally sampled batch_size for message
        passing reasons are not considered).
    mu:
        Mean of the negative binomial with positive support.
        (dim: batch_size x n_genes)
    theta:
        Inverse dispersion parameter with positive support.
        (dim: n_genes)
    eps:
        Numerical stability constant.

    Returns
    ----------
    nb_loss:
        Omics reconstruction loss using a negative binomial model.
    """
    theta = theta.expand_as(mu)  # Broadcast theta to match mu and x shape
    problematic_samples = (x.sum(1) == 0) & (mu.sum(1) == 0) & (theta.sum(1) == 0)
    
    x, mu, theta = x[~problematic_samples], mu[~problematic_samples], theta[~problematic_samples]
    
    if x.size(0) == 0:
        return torch.tensor(0.0, device=x.device)

    log_theta_mu_eps = torch.log(theta + mu + eps)
    log_likelihood_nb = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))
    nb_loss = torch.mean(-log_likelihood_nb.sum(-1))
    return nb_loss 

def compute_gp_l1_reg_loss(
        model: nn.Module,
        gp_type: Literal["prior", "addon"],
        l1_targets_mask: Optional[torch.Tensor]=None,
        l1_sources_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Compute L1 regularization loss for the rna decoder weights of gene programs
    of the type ´gp_type´ to encourage gene sparsity of those gene programs.

    Parameters
    ----------
    model:
        The VGPGAE module.
    gp_type:
        Type of gene programs to which the L1 regularization loss should be
        applied.
    l1_targets_mask:
        Boolean gene program gene mask that is True for all gene program target
        genes to which the L1 regularization loss should be applied (dim:
        n_genes, n_gps).
    l1_sources_mask:
        Boolean gene program gene mask that is True for all gene program source
        genes to which the L1 regularization loss should be applied (dim:
        n_genes, n_gps).

    Returns
    ----------
    gp_l1_reg_loss:
        L1 regularization loss for the rna decoder weights.
    """
    if gp_type == "prior":
        layer_name = "masked_l"
    elif gp_type == "addon":
        layer_name = "addon_l"

    # First compute layer-wise sum of absolute weights over target and source
    # rna decoder layers, then sum across layers. Use l1 masks to determine
    # which weights are included in the sum.
    # NOTE:
    # - the absolute weights and thus the L1 loss will be higher for highly
    #   expressed genes
    # - the model will keep weights non-zero for gps with very high scores and
    #   turn off weights for gps with low scores 

    # TODO: deal with the cases where masks are None
    # if l1_targets_mask is not None: # (N_genes, N_gp)
    #     l1_targets_mask = l1_targets_mask.unsqueeze(0) # (1, N_genes, N_gp)
    # if l1_sources_mask is not None: # (N_genes, N_gp)
    #     l1_sources_mask = l1_sources_mask.unsqueeze(0) # (1, N_genes, N_gp)

    # Calculate the overall L1 norm mean dynamically (depending on number of timepoints)
    decoder_layerwise_param_sum = torch.stack(
        [
            torch.mean(
                torch.linalg.vector_norm(
                    param[
                        l1_targets_mask.unsqueeze(0).expand(param.size(0), -1, -1) 
                        if "targets" in param_name else
                        l1_sources_mask.unsqueeze(0).expand(param.size(0), -1, -1)
                    ],
                    ord=1
                )
            )
            for param_name, param in model.named_parameters()
            if f"rna.nb_means_normalized_decoder.{layer_name}.weights" in param_name
        ],
        dim=0
    )

    gp_l1_reg_loss = torch.sum(decoder_layerwise_param_sum)
    return gp_l1_reg_loss