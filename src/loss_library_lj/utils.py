import torch
import torch.nn.functional as F

from typing import List

def get_entities(
        input: str,
        ner_model_lst: List, 
        linker_lst: List = None
        ):
    """Find and return a set of entities in an input string

    Args:
        input (str): Input string which will be parsed for entities
        ner_model_lst (List[spacy.ner]): List of Spacy NER models which will be used to
            find the named entities, models will be called using ner_model(s)
        linker_lst (List, optional): List of Spacy Linkers used (if applicable) to pick which entities to use. Defaults to None.

    Returns:
        set: Set of entities found within input
    """
    
    SEMTYPES = ["T023","T028","T046","T047","T048",
                "T059","T060","T061","T074","T109",
                "T116","T121","T122","T123","T125",
                "T129","T184","T191","T195"]

    output_entities = set()

    if type(ner_model_lst) is not list:
        ner_model_lst = [ner_model_lst]
        linker_lst    = [linker_lst]

    for (ner_model, linker) in zip(ner_model_lst, linker_lst):
        entity_lst = ner_model(input).ents

        if "scispacy_linker" in ner_model.pipe_names:
            filtered_entities = []
            for e in set(entity_lst):
                if len(e._.kb_ents) > 0:
                    umls_ent_id, _ = e._.kb_ents[0]  # Get top hit from UMLS
                    umls_ent  = linker.kb.cui_to_entity[umls_ent_id]  # Get UMLS entity
                    umls_semt = umls_ent[3]
                    if any([t in SEMTYPES for t in umls_semt]):
                        e = str(e)
                        if e not in filtered_entities:
                            filtered_entities.append(e)
            output_entities.update(set(filtered_entities))
        else:
            output_entities.update(set([str(e) for e in entity_lst]))

    return output_entities

def clean(s: str, tokens_to_remove: List[str]):
    """Replace all special tokens with the empty string

    Args:
        s (str): Original string
        tokens_to_remove: List[str]: List of tokens to remove (i.e. replace with empty string)

    Returns:
        str: Cleaned string
    """
    for t in tokens_to_remove:
        s = s.replace(str(t), "")
    return s

def compute_entity_mask(targets, tokenizer, ner_model, linker):
    """Create a tensor of shape [num_samples, vocab_size], where entry [i,j]
    is 1 if vocab_j is an entity and is present in the i-th sample

    Args:
        targets (torch.tensor): Tensor of IDs to look for entities from
        tokenizer (tokenizer): Tokenizer
        ner_model (List[spacy.ner]): List of Spacy NER models which will be used to
            find the named entities, models will be called using ner_model(s)
        linker (List[spacy.linker], optional): List of Spacy Linkers used (if applicable) 
            to pick which entities to use. Defaults to None.

    Returns:
        torch.tensor: Tensor of shape [num_samples, vocab_size]
    """

    # Decode and clean the targets
    decoded_tgt = tokenizer.batch_decode(targets) 
    decoded_tgt = [clean(s, tokenizer.all_special_tokens_extended) for s in decoded_tgt] 

    # Use NER model to identify entities in the targets
    entities    = [get_entities(s, ner_model, linker) for s in decoded_tgt] 
    entity_strs = [[str(s) for s in lst] for lst in entities] 

    # Create a tensor of shape [len(targets), vocab_size] 
    # Where entry (i,j) is 1 if vocab word vocab_j is an entity AND is present in target i 
    # (i.e. for each target, we get a vector marking which words in it are entities)
    entity_mask_lst = []
    for lst in entity_strs:
        if len(lst)>0:
            # Get the entities and tokenize them to get their IDs
            ent_toks = tokenizer(lst)
            # One hot encode to a vocab_size length vector, where it's 1 if the 
            # vocab word is present and is an entity
            sent_ent_mask = torch.zeros(tokenizer.vocab_size)
            sent_ent_mask[ent_toks] = 1.0
            # Save the vector
            entity_mask_lst.append(sent_ent_mask)
        else:
            entity_mask_lst.append(torch.zeros(tokenizer.vocab_size))
    entity_mask = torch.vstack(entity_mask_lst)

    # Remove any special IDs from the mask
    entity_mask[:, tokenizer.all_special_ids] = 0

    return entity_mask
