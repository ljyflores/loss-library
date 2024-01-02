import torch
import torch.nn.functional as F

from typing import List

def mutual_information_loss(
        logits: torch.tensor, 
        labels: torch.tensor, 
        model, 
        mi_weight: float, 
        filter: str = "all"
        ):
    """Adds an extra objective to maximize mutual information to the standard
    NLL loss; allows users to pick which tokens' mutual information to maximize,
    and returns the total loss (NLL loss + MI loss)

    Implementation of MI is taken from Fairseq implementation by van der Poel et al (citation)

    Args:
        logits (torch.tensor): Tensor of shape [batch_size, seq_len, vocab_size]
        labels (torch.tensor): Tensor of shape [batch_size, seq_len]
        model (model): Model
        mi_weight (float): Weight used for MI loss objective
        filter (str, optional): Which tokens' mutual information to use in the 
            maximization objective, either the target tokens ("labels"), the 
            generated tokens ("generated"), both target and generated ("both"),
            or all tokens ("all"). Defaults to "all".

    Returns:
        torch.tensor: Float value of MI loss across samples, returned in tensor for backpropagation
    """
    
    batch_size, seq_len, vocab_size = logits.shape

    # We're going to use the standard negative log likelihood as loss
    # Then ADD on a penalty that maximizes MI
    # Let's first compute NLL
    nll_loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        labels.view(-1), 
        reduction='none').mean(axis=-1).view(-1, batch_size)

    # Compute MI
    baseline_outputs = model(**{"input_ids": labels})
    baseline_logits  = baseline_outputs["logits"]
    mi_tensor = logits - baseline_logits

    # The filter parameter determines which tokens' MI to maximize
    # "labels" filters to tokens in the labels
    # "generated" filters to tokens in the generated tokens
    # "both" filters to both tokens in the labels and the generated tokens
    # "all" uses all the tokens' MIs
    
    if (filter == "generated") or (filter == "both"): 
        # Generate logits indices mask to determine generated words
        generated_indices = torch.argmax(logits, dim=-1)
        mask = torch.nn.functional.one_hot(
            generated_indices, num_classes=vocab_size
        ) 
    if (filter == "labels") or (filter == "both"):
        # Apply the MI loss only to the targets
        labels_indices = torch.nn.functional.one_hot(
            labels, 
            num_classes = vocab_size
        )
        mask = labels_indices
    if filter == "both":
        # Combine to get final target mask
        mask = 1*((generated_indices + labels_indices) > 0)
    if filter == "all":
        mask = torch.ones(batch_size, seq_len, vocab_size)

    # So, the MI tensor is the MI value x target_mask
    # Sum across the sequence length and vocab dimensions
    # We end up with an MI tensor of length batch_size
    mi = (mi_tensor * mask).sum(axis=-1).mean(axis=-1)

    # We want to MAXIMIZE the MI
    # We can do this by setting its loss to exp(-x)
    mi_loss = torch.exp(-1.0*mi)

    loss = nll_loss.mean(axis=-1) + (mi_weight * mi_loss)
    return loss