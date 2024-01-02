import torch
import torch.nn.functional as F

from typing import List
from utils import compute_entity_mask

def compute_rejection_loss(
    logits: torch.tensor,
    target: torch.tensor,
    epsilon: float,
    unk_idx: int,
    ignore_index: int = None,
    reduce: bool = True,
    mask: torch.tensor = None,
    alpha: float = 1.0
):
    """Computes rejection loss across samples
    Taken from implementation by Meng Cao (citation)

    Args:
        logits (torch.tensor): Tensor of shape [batch_size, seq_len, vocab_size]
        target (torch.tensor): Tensor of shape [batch_size, seq_len]
        epsilon (float): Weight to use on original NLL loss
        unk_idx (int): Index of <UNK> token
        ignore_index (int, optional): Index to ignore. Defaults to None.
        reduce (bool, optional): Whether or not to return full rejection loss tensor or average. Use True to return average.
        mask (torch.tensor, optional): If provided, it only computes the rejection loss 
            using vocab tokens for which mask is 1. This was used by Cao et al to filter
            the rejection loss only to entity tokens. Defaults to None.
        alpha (float, optional): Optional. Defaults to 1.0.

    Returns:
        (torch.tensor, torch.tensor): Tuple of rejection loss and orig NLL loss
    """
    batch_size, seq_len, vocab_size = logits.shape
    lprobs = torch.log(torch.softmax(logits, dim=-1))

    # if target.dim() == lprobs.dim() - 1:
    #     target = target.unsqueeze(-1)
    # nll_loss = -logits.gather(dim=-1, index=target)
    nll_loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        target.view(-1), 
        reduction='none'
        )
    nll_loss = nll_loss.view(batch_size, seq_len, 1)
    
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # ================== calculate rejection loss ==================
    # rej_prob = torch.exp(lprobs[:, unk_idx]).unsqueeze(-1) 

    # Modified here, get logit on the unk_token (3rd dim)
    rej_prob = torch.exp(lprobs[:, :, unk_idx]).unsqueeze(-1) 

    if mask is not None:
        # Take entity mask; unsqueeze from (batch_size x vocab_size)
        # to (batch_size x seq_len x vocab_size)
        mask = mask[:, :vocab_size].unsqueeze(1).expand(batch_size, seq_len, vocab_size).eq(0)
        # Mask out only in the targets 
        mask = mask * torch.nn.functional.one_hot(target, num_classes=vocab_size)
        # Squeeze into the seq_len dimension
        mask = mask.max(axis=-1, keepdim=True).values 
        # Turn mask into bool
        mask = mask == True
        # Mask the rej_probs
        keep_prob = (1. - rej_prob).masked_fill(mask, 1.0)  # 0: non-entity
    else:
        keep_prob = 1. - rej_prob
    assert keep_prob.shape == nll_loss.shape, \
        "nll_loss: {}; keep_prob: {}".format(nll_loss.shape, keep_prob.shape)    

    # Essentially masks out the tokens which we don't wanna keep
    # Done using the keep_prob logic, but instead of just 1 and 0,
    # keep_prob is continuous – hence it's a "soft" masking
    rej_loss = keep_prob * (nll_loss + torch.log(keep_prob))
    rej_regularizer = -alpha * torch.log(keep_prob)
    nll_loss = rej_loss + rej_regularizer

    rej_smooth_loss = keep_prob * (smooth_loss + torch.log(keep_prob))
    smooth_loss = rej_smooth_loss + rej_regularizer
    # ===============================================================

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.mean(axis=-1) # sum()
        smooth_loss = smooth_loss.mean(axis=-1) # sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def rejection_loss(
    logits: torch.tensor,
    labels: torch.tensor,
    tokenizer,
    ner_model: List,
    linker: List,
    ignore_index: int = None,
    reduce: bool = True,
    alpha: float = 1.0
):
    """Wrapper for rejection loss function, which first determines which tokens
    are entities, and uses this as a mask in the rejection loss function (i.e. goal
    is to compute the rejection loss only for entity tokens)

    Args:
        logits (torch.tensor): Tensor of shape [batch_size, seq_len, vocab_size]
        labels (torch.tensor): Tensor of shape [batch_size, seq_len]
        tokenizer (tokenizer): HF Tokenizer
        ner_model (List[spacy.ner]): List of Spacy NER models which will be used to
            find the named entities, models will be called using ner_model(s)
        linker (List[spacy.linker], optional): List of Spacy Linkers used (if applicable) 
            to pick which entities to use. Defaults to None.
        ignore_index (int, optional): Index to ignore. Defaults to None.
        reduce (bool, optional): Whether or not to return full rejection loss tensor or average. Use True to return average.
        alpha (float, optional): Optional. Defaults to 1.0.

    Returns:
        torch.tensor: Float value of rejection loss across samples, returned in tensor for backpropagation
    """
    # Compute entity_mask
    entity_mask = compute_entity_mask(
        targets = labels,
        tokenizer = tokenizer, 
        ner_model = ner_model,
        linker = linker)
    entity_mask = entity_mask.to(logits.device)

    loss, _ = rejection_loss(
        logits = logits,
        target = labels,
        epsilon = 0.1,
        unk_idx = tokenizer.unk_token_id,
        ignore_index = ignore_index, 
        reduce = reduce,
        mask = entity_mask,
        alpha = alpha
    )                
    return loss