# benchmark_ner.py – Python 3.8-compatible script for comparing NER models.
# Compares SJS/KL distillation vs. a non-distilled CRF baseline.
# Measures performance (F1) and efficiency (Time, TFLOPs, Memory).

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from filelock import FileLock
from seqeval.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
# Install prerequisites: pip install transformers seqeval torchcrf fvcore
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

# Setup logger
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

# =================================================================================
# 1. Data Loading & Processing Utilities ( Largely Unchanged )
# =================================================================================
# InputExample, InputFeatures, read_examples_from_file, etc. are identical to the original script.
# For brevity, they are assumed to be present here. The core changes are in the
# model definitions and the main benchmark runner logic.

@dataclass
class InputExample:
    """A single training/test example for token classification."""
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None

@dataclass
class InputFeatures:
    """A single set of features for an example."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]

def read_examples_from_file(data_dir: str, mode: str) -> List[InputExample]:
    """Reads a CoNLL-style text file."""
    file_path = os.path.join(data_dir, f"{mode}.tsv")
    examples = []
    words, labels, guid_idx = [], [], 1
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_idx}", words=words, labels=labels))
                    words, labels = [], []
                    guid_idx += 1
            else:
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1] if len(splits) > 1 else "O")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_idx}", words=words, labels=labels))
    return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Converts examples to features, handling subtokens."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex in examples:
        tokens, label_ids = [], []
        for word, label in zip(ex.words, ex.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens: word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            label_ids.extend([label_map[label]] + [IGNORE_INDEX] * (len(word_tokens) - 1))

        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]

        # Add special tokens
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [IGNORE_INDEX] + label_ids + [IGNORE_INDEX]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids) # Assuming single sequence

        # Pad
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [tokenizer.pad_token_type_id] * padding_length
        label_ids += [IGNORE_INDEX] * padding_length

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=label_ids))
    return features

class NerDataset(Dataset):
    def __init__(self, data_dir, tokenizer, labels, max_seq_length, mode, overwrite_cache=False):
        cached_features_file = os.path.join(data_dir, f"cached_{mode}_{max_seq_length}.pt")
        if os.path.exists(cached_features_file) and not overwrite_cache:
            self.features = torch.load(cached_features_file)
        else:
            examples = read_examples_from_file(data_dir, mode)
            self.features = convert_examples_to_features(examples, labels, max_seq_length, tokenizer)
            torch.save(self.features, cached_features_file)
    def __len__(self): return len(self.features)
    def __getitem__(self, i):
        f = self.features[i]
        return {
            "input_ids": torch.tensor(f.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(f.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(f.token_type_ids, dtype=torch.long),
            "labels": torch.tensor(f.label_ids, dtype=torch.long),
        }

def get_labels(path: str) -> List[str]:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return ["O", "B-Disposition", "I-Disposition", "B-NoDisposition", "I-NoDisposition", "B-Undetermined", "I-Undetermined"]


# =================================================================================
# 2. Argument Classes
# =================================================================================

@dataclass
class ModelArguments:
    """Arguments pertaining to which models/configs/tokenizers we are going to fine-tune from."""
    model_type: str = field(
        metadata={"help": "Type of student model to train: 'distill' or 'crf'."}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models for the teacher."}
    )
    student_config_name: Optional[str] = field(
        default=None, metadata={"help": "Path to a student config json file."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher."}
    )
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    data_dir: str = field(metadata={"help": "The input data dir. Should contain train, dev, and test files."})
    labels: Optional[str] = field(default=None, metadata={"help": "Path to a file containing all labels."})
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

@dataclass
class DistillArguments:
    """Arguments for distillation methods."""
    distillation_method: str = field(
        default="none", metadata={"help": "Distillation method: none | kl | sj | ssjs"}
    )
    temperature: float = field(default=2.0)
    alpha_ce: float = field(default=0.5, metadata={"help": "Weight of CE loss vs. distillation loss."})
    beta_mse: float = field(default=0.1, metadata={"help": "Weight of hidden-state MSE loss."})
    lambda_sjs: float = field(default=1.0, metadata={"help": "Weight of transition-JS loss in SJS."})


# =================================================================================
# 3. Model Architectures (Distill Student & CRF Student)
# =================================================================================

class StudentConfig(PretrainedConfig):
    model_type = "student"
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=2, num_attention_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

class StudentModelForTokenClassification(PreTrainedModel):
    """Standard student model for distillation."""
    config_class = StudentConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embedding_output = self.embeddings(input_ids)
        key_padding_mask = attention_mask == 0
        encoder_output = self.encoder(embedding_output, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(encoder_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=(encoder_output,))


class StudentModelWithCRF(PreTrainedModel):
    """Student model with a CRF layer on top. Not for distillation."""
    config_class = StudentConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.student_base = StudentModelForTokenClassification(config) # Re-use the base
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Get emissions from the base student model
        outputs = self.student_base(input_ids=input_ids, attention_mask=attention_mask)
        emissions = outputs.logits
        mask = attention_mask.bool()

        loss, logits = None, None
        if labels is not None:
            # Training: CRF calculates the log-likelihood loss
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
        
        # Inference: CRF decodes the best sequence
        decoded_sequences = self.crf.decode(emissions, mask=mask)
        
        # Pad decoded sequences to be of the same size for Trainer compatibility
        max_len = emissions.shape[1]
        padded_logits = torch.full((len(decoded_sequences), max_len), IGNORE_INDEX)
        for i, seq in enumerate(decoded_sequences):
            padded_logits[i, :len(seq)] = torch.tensor(seq)

        return TokenClassifierOutput(loss=loss, logits=padded_logits)

# =================================================================================
# 4. Distillation Trainer & Helpers
# =================================================================================

def js_divergence(p, q, eps=1e-6):
    m = 0.5 * (p + q)
    return 0.5 * (nn.functional.kl_div(torch.log(m+eps), p, reduction='none') + 
                   nn.functional.kl_div(torch.log(m+eps), q, reduction='none')).sum(-1)

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, distill_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.to(self.args.device).eval()
        self.distill_args = distill_args
        self.proj = None
        if self.model.config.hidden_size != self.teacher.config.hidden_size:
            self.proj = nn.Linear(self.model.config.hidden_size, self.teacher.config.hidden_size).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # For memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Student forward pass
        student_outputs = model(**inputs, output_hidden_states=True)
        ce_loss = student_outputs.loss

        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)

        # Masking to only compute loss on valid tokens
        mask = inputs["labels"] != IGNORE_INDEX
        
        # --- Distillation Loss ---
        distill_loss = 0.0
        T = self.distill_args.temperature
        method = self.distill_args.distillation_method
        
        s_logits, t_logits = student_outputs.logits[mask], teacher_outputs.logits[mask]
        
        if method == 'kl':
            kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(s_logits / T, dim=-1),
                nn.functional.softmax(t_logits / T, dim=-1),
                reduction='batchmean'
            ) * (T ** 2)
            distill_loss = kl_loss
        
        elif method in ['sj', 'ssjs']:
            s_probs, t_probs = nn.functional.softmax(s_logits, -1), nn.functional.softmax(t_logits, -1)
            token_js = js_divergence(s_probs, t_probs).mean()
            
            if method == 'sj':
                distill_loss = token_js
            else: # ssjs
                s_full_probs = nn.functional.softmax(student_outputs.logits, -1)
                t_full_probs = nn.functional.softmax(teacher_outputs.logits, -1)
                
                # Outer product for transition matrix
                s_trans = torch.einsum('bti,btj->btij', s_full_probs[:, :-1, :], s_full_probs[:, 1:, :])
                t_trans = torch.einsum('bti,btj->btij', t_full_probs[:, :-1, :], t_full_probs[:, 1:, :])
                
                trans_mask = (inputs["labels"][:, :-1] != IGNORE_INDEX) & (inputs["labels"][:, 1:] != IGNORE_INDEX)
                trans_js = js_divergence(s_trans[trans_mask], t_trans[trans_mask]).mean()
                distill_loss = token_js + self.distill_args.lambda_sjs * trans_js

        # --- Hidden State MSE Loss ---
        mse_loss = 0.0
        s_hidden = student_outputs.hidden_states[-1]
        t_hidden = teacher_outputs.hidden_states[-1]
        if self.proj:
            s_hidden = self.proj(s_hidden)
        mse_loss = nn.functional.mse_loss(s_hidden[mask], t_hidden[mask])

        # --- Final Loss ---
        a, b = self.distill_args.alpha_ce, self.distill_args.beta_mse
        total_loss = a * ce_loss + (1 - a) * distill_loss + b * mse_loss

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                self.log({"peak_gpu_memory_mb": peak_mem})

        return (total_loss, student_outputs) if return_outputs else total_loss


# =================================================================================
# 5. Metrics & Prediction Alignment
# =================================================================================

def align_predictions(predictions, label_ids, id2label, is_crf=False):
    preds_list, labels_list = [], []
    if is_crf:
        preds = predictions
    else:
        preds = np.argmax(predictions, axis=2)

    for i in range(preds.shape[0]):
        pred_row, label_row = [], []
        for j in range(preds.shape[1]):
            if label_ids[i, j] != IGNORE_INDEX:
                pred_row.append(id2label[preds[i, j]])
                label_row.append(id2label[label_ids[i, j]])
        preds_list.append(pred_row)
        labels_list.append(label_row)
    return preds_list, labels_list

def build_compute_metrics(id2label, is_crf=False):
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        preds, golds = align_predictions(p.predictions[0], p.label_ids, id2label, is_crf)
        report = classification_report(golds, preds, output_dict=True, zero_division=0)
        return {"f1": report["micro avg"]["f1-score"], "precision": report["micro avg"]["precision"], "recall": report["micro avg"]["recall"]}
    return compute_metrics

# =================================================================================
# 6. Main Benchmark Runner
# =================================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DistillArguments, TrainingArguments))
    m_args, d_args, dist_args, t_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    set_seed(t_args.seed)

    labels = get_labels(d_args.labels)
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(labels)
    
    t_args.remove_unused_columns = False # Important for custom models

    # --- Tokenizer & Teacher (if distilling) ---
    tokenizer = AutoTokenizer.from_pretrained(m_args.tokenizer_name or m_args.teacher_model_name_or_path, use_fast=True)
    teacher = None
    if m_args.model_type == 'distill':
        teacher_config = AutoConfig.from_pretrained(m_args.teacher_model_name_or_path, num_labels=num_labels, output_hidden_states=True)
        teacher = AutoModelForTokenClassification.from_pretrained(m_args.teacher_model_name_or_path, config=teacher_config)

    # --- Student Model Selection ---
    student_config = StudentConfig.from_pretrained(m_args.student_config_name, num_labels=num_labels) if m_args.student_config_name else StudentConfig(num_labels=num_labels, vocab_size=tokenizer.vocab_size)
    
    student = None
    if m_args.model_type == 'crf':
        logger.info("Initializing Student model with CRF layer.")
        student = StudentModelWithCRF(student_config)
    elif m_args.model_type == 'distill':
        logger.info(f"Initializing Student model for {dist_args.distillation_method} distillation.")
        student = StudentModelForTokenClassification(student_config)
    else:
        raise ValueError("Invalid model_type. Choose 'distill' or 'crf'.")

    # --- Datasets ---
    train_ds = NerDataset(d_args.data_dir, tokenizer, labels, d_args.max_seq_length, "train_dev", d_args.overwrite_cache) if t_args.do_train else None
    dev_ds = NerDataset(d_args.data_dir, tokenizer, labels, d_args.max_seq_length, "devel", d_args.overwrite_cache) if t_args.do_eval else None
    
    # --- Trainer Selection ---
    is_crf = m_args.model_type == 'crf'
    compute_metrics_fn = build_compute_metrics(id2label, is_crf=is_crf)
    
    trainer = None
    if is_crf:
        trainer = Trainer(model=student, args=t_args, train_dataset=train_ds, eval_dataset=dev_ds, compute_metrics=compute_metrics_fn)
    else:
        trainer = DistillationTrainer(model=student, teacher_model=teacher, distill_args=dist_args, args=t_args, train_dataset=train_ds, eval_dataset=dev_ds, compute_metrics=compute_metrics_fn)
    
    # --- Training & Wall-Clock Time ---
    if t_args.do_train:
        logger.info(f"*** Starting Training for model: {m_args.model_type}, method: {dist_args.distillation_method if not is_crf else 'CRF'} ***")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        trainer.save_model()
        tokenizer.save_pretrained(t_args.output_dir)
        # Log training time
        trainer.log({"training_time_seconds": training_time})

    # --- Evaluation ---
    if t_args.do_eval:
        logger.info("*** Evaluating Student ***")
        metrics = trainer.evaluate()
        logger.info(f"Student F1: {metrics['eval_f1']:.4f}")

    # --- TFLOPs Analysis (Inference) ---
    try:
        from fvcore.nn import FlopCountAnalysis
        logger.info("*** Analyzing Inference TFLOPs ***")
        # Get a single batch for analysis
        data_loader = DataLoader(dev_ds, batch_size=t_args.per_device_eval_batch_size)
        batch = next(iter(data_loader))
        batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
        # For FLOPs, we analyze the forward pass without labels
        batch.pop("labels")

        flops = FlopCountAnalysis(student, batch)
        total_flops = flops.total()
        logger.info(f"Inference FLOPs per batch: {total_flops / 1e9:.4f} GFLOPs")
        if t_args.do_train:
             trainer.log({"inference_gflops_per_batch": total_flops / 1e9})
    except ImportError:
        logger.warning("fvcore not installed. Skipping TFLOPs analysis. `pip install fvcore`")

    logger.info("Benchmark run finished.")


if __name__ == "__main__":
    main()