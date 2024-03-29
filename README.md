# Overview
<b>Loss Library</b> aims to help users plug-and-play loss functions from various NLP papers. We invite contributions to the package by adding loss functions!



## Supported Losses
| Loss Function | Description / Intuition | Citation |
| ------------- | ----------------------- | -------- | 
| Unlikelihood Loss            | Penalize unwanted tokens, based on user-provided weights | (<a href="https://arxiv.org/abs/1908.04319">Welleck et al., 2019</a>) |
| Rejection Loss               | Penalize models for being uncertain, by letting it add probability on `<UNK>` token, and reduce probability from unsure tokens | (<a href="https://aclanthology.org/2022.emnlp-main.663.pdf">Cao et al., 2022</a>) |
| Mutual Information           | Penalize models for being uncertain, by maximizing the entropy between the predicted and baseline prediction | Inspired by (<a href="https://arxiv.org/abs/2210.13210">Van der Poel et al., 2022</a>) |
| Loss Truncation              | Remove noisy training examples, based on the heuristic that noisy examples have higher cross-entropy loss | (<a href="https://aclanthology.org/2020.acl-main.66/">Kang and Hashimoto, 2020</a>) |
| Mutual-Info Based Loss Truncation | Remove noisy training examples, where noisy examples have very different predictions between baseline and predicted | Inspired by (<a href="https://aclanthology.org/2020.acl-main.66/">Kang and Hashimoto, 2020</a>) |
| Entity-Level Loss Truncation | Remove noisy training examples, where we use the cross-entropy loss of <i>only</i> the entity tokens | Inspired by (<a href="https://aclanthology.org/2020.acl-main.66/">Kang and Hashimoto, 2020</a>) |

# Set-Up
<b>Loss Library</b> currently supports Python 3.8, and can be loaded in as follows:
```
git clone https://github.com/ljyflores/loss-library.git
cd loss-library
pip install .
```

# Usage
To use Loss Library, we instantiate `LossLibrary` using the desired loss function, and call it to compute the loss at each training step.

## Without `Trainer`
```python
import loss_library

# Instantiate loss function
loss_function = loss_library.LossLibrary(
    loss_type = "ul",
    tokenizer = trainer.tokenizer,
    model = trainer.model, 
    ul_weights_path = f"{ROOT_PATH}/fk_weights.pkl",
    ul_lambda_read  = 5e-4,
    ul_lambda_const = 1.5e-4,
    ul_check_input_ents = True,
    ul_check_label_ents = True
)

# Run training loop
model.train()

for idx, batch in enumerate(dataloader_train):
    # Unpack batch
    labels = batch.pop("labels")
    inputs = {key:batch[key].cuda() for key in batch}
    labels = labels.cuda()

    # Forward pass
    optimizer.zero_grad()
    outputs = model(**inputs)
    logits = outputs.get("logits")
            
    # Compute loss
    loss = loss_function(
              logits = logits, 
              labels = labels, 
              inputs = inputs
            )
    # Backpropagate and step
    loss.backward()    
    optimizer.step()
    scheduler.step()
```

## With `Trainer`
When using `Trainer`, we add `LossLibrary` as an attribute of the Trainer. 
We use it to compute loss by subclassing `Trainer` and changing the `compute_loss` function.

Step 1: Subclass `Trainer` and change the `compute_loss` function
```python
class SimplificationTrainer(Seq2SeqTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models
        return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
            
        # Unpack the logits and labels
        logits = outputs["logits"]
        labels = inputs["labels"].to(logits.device)

        # Compute the loss using LossLibrary, which is stored in self.loss_function
        loss = self.loss_function(
            logits = logits, 
            labels = labels, 
            inputs = inputs
        )
        return (loss, outputs) if return_outputs else loss
```

Step 2: Add `LossLibrary` to the Trainer
```python
import loss_library

# Initialize the model (using the model_init_func)
def model_init_func(trial):
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    return model

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

# Set up Trainer
training_args = Seq2SeqTrainingArguments(...)
data_collator = DataCollatorForSeq2Seq(tokenizer)
trainer = SimplificationTrainer(
    model_init = model_init_func,  
    args = training_args,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    data_collator = data_collator,
    tokenizer = tokenizer
)

# Add the LossLibrary loss function to the trainer
trainer.loss_function = loss_library.LossLibrary(
    loss_type = "ul",
    tokenizer = trainer.tokenizer,
    model = trainer.model, 
    ul_weights_path = f"{ROOT_PATH}/fk_weights.pkl",
    ul_lambda_read  = 5e-4,
    ul_lambda_const = 1.5e-4,
    ul_check_input_ents = True,
    ul_check_label_ents = True
)
```

# Parameters
All losses will require three parameters: the tokenizer (`tokenizer`), model (`model`), and loss type (`loss_type`).

In addition, we specify other loss-specific parameters.

### Unlikelihood Loss
| Parameter | Values | Description |
| - | - | - |
| `model` | model | | 
| `tokenizer` | tokenizer | | 
| `loss_type` | `"ul"` | | 
| `ul_weights_path` | str | Path to a pkl file, which contains a list of float values which correspond to the penalties to apply to each vocab token |
| `ul_lambda_read` | float (default: `5e-4`) | Weight on the readability penalty (which is added to NLL loss) |
| `ul_lambda_const` | float (default: `1.5e-4`) | Weight on the consistency penalty (which is added to NLL loss) |
| `ul_check_input_ents` | bool (default: `True`) | Whether or not to penalize generated entities that are unsupported by the input |
| `ul_check_label_ents` | bool (default: `True`) | Whether or not to penalize generated entities that are unsupported by the label |

### Rejection Loss
| Parameter | Values | Description |
| - | - | - |
| `model` | model | | 
| `tokenizer` | tokenizer | | 
| `loss_type` | `"rej"` | | 

### Mutual Information Loss
| Parameter | Values | Description |
| - | - | - |
| `model` | model | | 
| `tokenizer` | tokenizer | | 
| `loss_type` | `"mi"` | | 
| `mi_weight` | float (default: `1.0`) | Weight on the mutual information penalty (which is added to NLL loss) |
| `mi_filter` | str (default: `"all"`) | Which tokens' entities to maximize mutual information for, either `"generated"` for the tokens which the model would generate (i.e. have the highest probabilities at time of generation), `"labels"` for the tokens in the label, `"both"` for the generated and label tokens, or `"all"` for all tokens |

### Loss Truncation
| Parameter | Values | Description |
| - | - | - |
| `model` | model | | 
| `tokenizer` | tokenizer | | 
| `loss_type` | `"lt"`/`"mi_lt"`/`"max_lt"` | `"lt"` for LT using standard NLL loss, `"mi_lt"` for mutual-info based, and `"max_lt"` for entity-based | 
| `lt_dropc` | float (default=`0.4`) | The percent of training examples to drop from training |
| `lt_min_count` | int (default=`500`) | The number of examples trained on before the cutoff is computed |
| `lt_recompute` | int (default=`500`) | The number of training steps taken before the loss cutoff is recomputed | 
