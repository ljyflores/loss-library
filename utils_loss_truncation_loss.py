import torch
import torch.nn.functional as F

from typing import List
from utils import compute_entity_mask

def loss_truncation(
        logits: torch.tensor, 
        labels: torch.tensor, 
        model, 
        tokenizer, 
        ner_model, 
        linker, 
        loss_dropper, 
        lt_type: str = "lt"
        ):
    """Wrapper for LossDropper implementation of loss truncation, computes the score
    which determines which examples to drop, and feeds them to the LossDropper instance
    to implement loss truncation; returns the loss from loss truncation

    Args:
        logits (torch.tensor): Tensor of shape [batch_size, seq_len, vocab_size]
        labels (torch.tensor): Tensor of shape [batch_size, seq_len]
        model (model): Model
        tokenizer (tokenizer): HF Tokenizer
        ner_model (List[spacy.ner]): List of Spacy NER models which will be used to
            find the named entities, models will be called using ner_model(s)
        linker (List[spacy.linker], optional): List of Spacy Linkers used (if applicable) 
            to pick which entities to use. Defaults to None.
        loss_dropper (LossDropper): LossDropper instance
        lt_type (str, optional): Score to use for loss truncation, either "mi_lt" to use
            mutual information, "max_lt" for entity-based NLL, or "lt" for plain NLL (orig
            definition). Defaults to "lt".

    Returns:
        torch.tensor: Float value of truncated loss across samples, returned in tensor for backpropagation
    """

    batch_size, seq_len, vocab_size = logits.shape

    # Loss truncation always uses the original negative log likelihood as loss
    # So let's first compute that here
    nll_loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        labels.view(-1), 
        reduction='none').mean(axis=-1).view(-1, batch_size)
    
    # All loss truncation is doing is removing examples from the training procedure
    # (i.e. by setting NLL to 0) according to some score. We now compute that score,
    # depending on which type of LT we're using.

    # If we're using entity-based loss truncation (max_lt), the score is computed as the
    # sum of NLL of only the entity tokens
    if lt_type == "max_lt":

        # Identify which tokens are entities in labels
        entity_mask = compute_entity_mask(labels, tokenizer, ner_model, linker).to(logits.device)
        entity_mask = entity_mask[:, :vocab_size] 
        entity_mask = entity_mask.unsqueeze(1)\
                        .expand(batch_size, seq_len, vocab_size) 

        label_mask = torch.nn.functional.one_hot(
            labels, 
            num_classes=vocab_size
            )
        
        # Create the mask which is 1 if a token is an entity
        # and 0 otherwise
        label_entity_mask = entity_mask * label_mask
        label_entity_mask = label_entity_mask.amax(axis=-1)

        # Change: Sum the loss across all the entities
        # max_nll = (nll_loss * label_entity_mask).amax(-1)
        max_nll = nll_loss * label_entity_mask
        max_nll = max_nll.sum(axis=-1) # / (1*(max_nll > 0)).sum(axis=-1)
        score = max_nll

    elif lt_type == "mi_lt":
    
        # Compute baseline logits
        baseline_outputs = model(**{"input_ids": labels})
        baseline_logits  = baseline_outputs["logits"][:seq_len, :]
        baseline_logits = baseline_logits.unsqueeze(0).expand(
            batch_size, seq_len, vocab_size
            )
        
        mi = logits - baseline_logits
        # mi = mi * entity_mask * label_mask
        mi = mi.sum(axis=-1)
        # mi = mi.sum(axis=-1) / (1*(mi > 0)).sum(axis=-1)
        score = mi
        
    # Otherwise, use the original NLL loss as the score
    else:
        score = nll_loss

    # Apply loss truncation: Find examples with a bad (i.e. high) score and mask them out
    mask = loss_dropper(score)
    loss = loss * mask  # Mask out the high losses
    loss = loss.mean()  # Aggregate
    return loss