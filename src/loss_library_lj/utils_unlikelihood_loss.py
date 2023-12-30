import torch
import torch.nn.functional as F

from typing import List
from utils import compute_entity_mask

def compute_unlikelihood_loss(
    decoder_input_ids: torch.tensor, 
    logits: torch.tensor, 
    weight_mask: torch.tensor
    ):        
    """Computes the total unlikelihood loss across the examples
    Taken from LuJunru NAPSS paper: https://github.com/LuJunru/NapSS/blob/main/modeling/finetune.py
    Under the hood, the unlikelihood loss for a given token is:
        (1-prob_of_generating_token) x (1 if token is in the label else 0) x (weight_on_vocab_token)
    Hence, we want to penalize unlikely tokens (hence if prob is low, 1-prob is high), in accordance with a
        weight that we provide, but only if it's in the label

    Args:
        decoder_input_ids (torch.tensor): Tensor of shape [batch_size, seq_length]
        logits (torch.tensor): Tensor of shape [batch_size, seq_length, vocab_size]
        weight_mask (torch.tensor): Tensor of shape [batch_size, seq_length, vocab_size]

    Returns:
        torch.tensor: Float value of unlikelihood loss across samples, returned in tensor for backpropagation
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    neg_probs = 1 - probs

    # replace zeros with small positive constant for stability
    neg_probs += (neg_probs == 0).float() * 1e-8
    log_neg_probs = torch.log(neg_probs)  # (N,s,v)

    # now create attention mask and apply it
    attention_mask = decoder_input_ids.eq(1).eq(0).float()
    attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, logits.shape[2])
    log_neg_probs_masked = log_neg_probs * attention_mask

    # apply weight vector to the log probability tensor
    weighted_probs = log_neg_probs_masked * weight_mask

    return -1.0 * weighted_probs.sum(axis=-1).sum(axis=-1)

def unlikelihood_loss(
        logits, 
        inputs, 
        tokenizer, 
        ner_model, 
        linker, 
        ul_weights, 
        lambda_read = 1.0, 
        lambda_const = 1e-3, 
        exclude_entities = False, 
        selective_penalty = True,
        readability_penalty = True,
        use_input_ents_to_check_hallucination = True,
        use_label_ents_to_check_hallucination = True
    ):
    """Wrapper for the unlikelihood loss function, which computes how much to penalize 
    each token (i.e. it computes the ul_weight), and feeds this into the UL function

    Args:
        logits (_type_): _description_
        inputs (_type_): _description_
        tokenizer (_type_): _description_
        ner_model (_type_): _description_
        linker (_type_): _description_
        ul_weights (_type_): _description_
        lambda_read (float, optional): _description_. Defaults to 1.0.
        lambda_const (_type_, optional): _description_. Defaults to 1e-3.
        exclude_entities (bool, optional): _description_. Defaults to False.
        selective_penalty (bool, optional): _description_. Defaults to True.
        readability_penalty (bool, optional): _description_. Defaults to True.
        use_input_ents_to_check_hallucination (bool, optional): _description_. Defaults to True.
        use_label_ents_to_check_hallucination (bool, optional): _description_. Defaults to True.

    Returns:
        torch.tensor: Float value of unlikelihood loss across samples, returned in tensor for backpropagation
    """

    batch_size, seq_len, vocab_size = logits.shape

    # Compute cross entropy loss
    labels = labels.to(logits.device)
    nll_loss = F.cross_entropy(
        logits.view(-1, vocab_size), 
        labels.view(-1), 
        reduction='none')

    # Generate UL weights
    ul_weight = torch.zeros((batch_size, 
                            seq_len, 
                            vocab_size)).float().cuda()

    # Create entity mask if we are excluding entities
    if exclude_entities:
        # Generate entity mask (bs x seq_len x vocab_size), which is 1 if
        # the token is an entity present in the INPUTS, 0 otherwise

        # Identify which tokens are entities in both labels and generated
        entity_mask = compute_entity_mask(
            inputs["input_ids"], tokenizer, ner_model, linker).to(logits.device)

        # Cut to vocab size
        entity_mask = entity_mask[:, :vocab_size] 

        # Expand across sequence length dimension
        entity_mask = entity_mask.unsqueeze(1)\
                        .expand(batch_size, seq_len, vocab_size) 
        
    # Selectivity Option
    if selective_penalty:
        # Generate logits indices mask to determine generated words
        # This is called selective penalty in the NAPSS paper
        logits_indices = torch.argmax(logits, dim=-1)
        logits_indices_mask = torch.nn.functional.one_hot(
            logits_indices, num_classes=vocab_size
        )  # (N,s,v)
    else:
        logits_indices_mask = torch.ones(
            batch_size, seq_len, vocab_size
        )

    # Readability Penalty
    # Applying this penalizes ONLY the generated words by their complexity
    if readability_penalty:
        read_mask = ul_weights
        read_mask = (
            read_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, seq_len, vocab_size)
            .clone()
        )

        if exclude_entities:
            # Find tokens which are NOT entities and keep the UL penalty on those
            # Zero out the penalty on entity words
            ul_weight += lambda_read * read_mask * logits_indices_mask * 1*(entity_mask.eq(0))
        else:
            ul_weight += lambda_read * read_mask * logits_indices_mask

    # Hallucination Penalty
    hallucination_penalty = []
    if use_input_ents_to_check_hallucination:
        hallucination_penalty.append("inputs")
    if use_label_ents_to_check_hallucination:
        hallucination_penalty.append("labels")

    if hallucination_penalty:
        hall_mask = (
            torch.zeros((batch_size, seq_len, vocab_size)).float().cuda()
        )
        if "labels" in hallucination_penalty:
            # This is a penalty on words which are not in the labels
            # This reduces hallucination
            labels_indices_mask = torch.nn.functional.one_hot(
                labels, num_classes=vocab_size
            )
            hall_mask += (
                labels_indices_mask.sum(axis=1)
                .unsqueeze(1)
                .expand(batch_size, seq_len, vocab_size)
            )
        if "inputs" in hallucination_penalty:
            inputs_indices_mask = torch.nn.functional.one_hot(
                inputs["input_ids"], num_classes=vocab_size
            )
            hall_mask += (
                inputs_indices_mask.sum(axis=1)
                .unsqueeze(1)
                .expand(batch_size, seq_len, vocab_size)
            )
        # Get the tokens which do not appear in either label or input
        neg_indices_mask = 1.0 * (hall_mask == 0) 
        if exclude_entities:
            # Find tokens which ARE entities and keep the hallucination penalty on those
            # Zero out non-entity words
            ul_weight += lambda_const * neg_indices_mask * logits_indices_mask * 1*(entity_mask.eq(1))
        else:
            # Penalize these non-appearing tokens with a fixed weight
            ul_weight += lambda_const * neg_indices_mask * logits_indices_mask

    ul_loss = compute_unlikelihood_loss(
        decoder_input_ids=labels, logits=logits, weight_mask=ul_weight
    )

    # Add in the UL loss to the orig loss
    loss = nll_loss.mean(axis=-1) + ul_loss
    return loss