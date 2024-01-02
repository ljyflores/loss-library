import spacy
import scispacy
import pickle

from scispacy.linking import EntityLinker
from utils_loss_truncation_loss import loss_truncation_loss, LossDropper
from utils_mutual_info_loss import mutual_information_loss
from utils_rejection_loss import rejection_loss
from utils_unlikelihood_loss import unlikelihood_loss

class LossLibrary:
    def __init__(
            self, 
            loss_type, 
            tokenizer,
            model,
            ul_weights_path = None,
            ul_lambda_read  = 5e-4,
            ul_lambda_const = 1.5e-4,
            ul_check_input_ents = True,
            ul_check_label_ents = True,
            lt_dropc = 0.4,
            lt_min_count = 500,
            lt_recompute = 500,
            mi_weight = 1.0,
            mi_filter = "all"
        ):

        self.loss_type = loss_type 
        self.tokenizer = tokenizer
        self.model     = model

        # Add loss specific arguments to the trainer
        if loss_type == "ul":
            # Import the weights
            with open(ul_weights_path, "rb") as f:
                ul_weights = pickle.load(f)
            ul_weights = list(map(lambda x: max(x, 0.0), ul_weights))
            self.ul_weights = ul_weights
            self.ul_lambda_read  = ul_lambda_read
            self.ul_lambda_const = ul_lambda_const

            # Import NER models and linkers
            ner_model_web = spacy.load("en_core_web_lg")
            ner_model_sci = spacy.load("en_core_sci_lg")
            ner_model_sci.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker_sci = ner_model_sci.get_pipe("scispacy_linker")

            # Specify which entities to use for hallucination checking
            self.ul_check_input_ents = ul_check_input_ents
            self.ul_check_label_ents = ul_check_label_ents

            # Add the NER models and linkers to the clsass
            self.ner_model = [ner_model_sci, ner_model_web]
            self.linker    = [linker_sci, None]

        if loss_type == "rej":
            # Import NER models and linkers
            self.ner_model = [spacy.load("en_core_web_lg")]
            self.linker    = [None]

        if loss_type == "max_lt":
            # Import NER models and linkers
            ner_model_web = spacy.load("en_core_web_lg")
            ner_model_sci = spacy.load("en_core_sci_lg")
            ner_model_sci.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
            )
            linker_sci = ner_model_sci.get_pipe("scispacy_linker")

            # Add them to the class
            self.ner_model = [ner_model_sci, ner_model_web]
            self.linker    = [linker_sci, None]

        if loss_type in ["lt","max_lt","mi_lt"]:
            self.loss_dropper = LossDropper(
                dropc=lt_dropc, 
                min_count=lt_min_count, 
                recompute=lt_recompute, 
                verbose=True
                )

        if loss_type == "mi":
            self.mi_weight = mi_weight
            self.filter = mi_filter 
    
    def forward(self, logits, labels, inputs=None):
        if self.loss_type == "rej":
            loss = rejection_loss(
                logits = logits, 
                labels = labels, 
                tokenizer = self.tokenizer, 
                ner_model = self.ner_model, 
                linker = self.linker
                )

        if self.loss_type == "ul":
            loss = unlikelihood_loss(
                logits, 
                inputs, 
                tokenizer = self.tokenizer, 
                ner_model = self.ner_model, 
                linker = self.linker, 
                ul_weights = self.ul_weights, 
                lambda_read = self.ul_lambda_read, 
                lambda_const = self.ul_lambda_const,
                exclude_entities = False, 
                selective_penalty = True,
                readability_penalty = True,
                use_input_ents_to_check_hallucination = self.ul_check_input_ents,
                use_label_ents_to_check_hallucination = self.ul_check_label_ents
            )
    
        if self.loss_type in ["lt", "max_lt", "mi_lt"]:
            loss = loss_truncation_loss(
                logits = logits, 
                labels = labels, 
                model = self.model, 
                tokenizer = self.tokenizer, 
                ner_model = self.ner_model, 
                linker = self.linker, 
                loss_dropper = self.loss_dropper, 
                lt_type = self.loss_type
                )

        if self.loss_type == "mi":
            loss = mutual_information_loss(
                logits = logits, 
                labels = labels, 
                model = self.model, 
                mi_weight = self.mi_weight, 
                filter=self.filter
                )
        return loss