# performance_analysis.py – Python 3.8-compatible script for comparing NER models.
# Compares SJS/KL distillation vs. a non-distilled CRF baseline.
# Measures performance (F1) and efficiency (Training Time, Eval Time, Inference Time, TFLOPs, Memory).

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from filelock import FileLock
from seqeval.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
# Install prerequisites: pip install transformers seqeval torchcrf fvcore accelerate
from torchcrf import CRF
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
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
# 1. Data Loading & Processing Utilities
# =================================================================================

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
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}. Skipping this dataset.")
        return []
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

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        label_ids = [IGNORE_INDEX] + label_ids + [IGNORE_INDEX]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [tokenizer.pad_token_type_id] * padding_length
        label_ids += [IGNORE_INDEX] * padding_length

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label_ids=label_ids))
    return features

class NerDataset(Dataset):
    def __init__(self, data_dir, tokenizer, labels, max_seq_length, mode, overwrite_cache=False):
        self.features = []
        examples = read_examples_from_file(data_dir, mode)
        if not examples:
            return
            
        cached_features_file = os.path.join(data_dir, f"cached_{mode}_{tokenizer.__class__.__name__}_{max_seq_length}.pt")
        lock_file = cached_features_file + ".lock"
        with FileLock(lock_file):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                self.features = torch.load(cached_features_file, weights_only=False)
            else:
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

def get_labels(path: Optional[str] = None) -> List[str]:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return ["O", "B-Disposition", "I-Disposition", "B-NoDisposition", "I-NoDisposition", "B-Undetermined", "I-Undetermined"]

# =================================================================================
# 2. Argument Classes
# =================================================================================

@dataclass
class ModelArguments:
    model_type: str = field(
        metadata={"help": "Type of student model to train: 'distill' or 'crf'."}
    )
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier for the teacher. Required for 'distill' model_type."}
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
    data_dir: str = field(metadata={"help": "The input data dir. Should contain train_dev.tsv, devel.tsv, and test.tsv."})
    labels: Optional[str] = field(default=None, metadata={"help": "Path to a file containing all labels."})
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

@dataclass
class DistillArguments:
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
    config_class = StudentConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, **kwargs):
        embedding_output = self.embeddings(input_ids)
        key_padding_mask = attention_mask == 0
        encoder_output = self.encoder(embedding_output, src_key_padding_mask=key_padding_mask)
        logits = self.classifier(encoder_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=(encoder_output,) if output_hidden_states else None)

class StudentModelWithCRF(PreTrainedModel):
    config_class = StudentConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.student_base = StudentModelForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.student_base(input_ids=input_ids, attention_mask=attention_mask)
        emissions = outputs.logits
        loss = None
        if labels is not None:
            mask = attention_mask.bool() if attention_mask is not None else None
            crf_labels = labels.clone()
            crf_labels[crf_labels == IGNORE_INDEX] = 0
            loss = -self.crf(emissions, crf_labels, mask=mask, reduction='mean')
        return TokenClassifierOutput(loss=loss, logits=emissions)

# =================================================================================
# 4. Trainers & Helpers
# =================================================================================

def js_divergence(p, q, eps=1e-6):
    m = 0.5 * (p + q)
    return 0.5 * (nn.functional.kl_div(torch.log(m + eps), p, reduction='none') +
                   nn.functional.kl_div(torch.log(m + eps), q, reduction='none')).sum(-1)

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, distill_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.args.device).eval()
        self.distill_args = distill_args
        self.proj = None
        if self.teacher is not None and hasattr(self.model.config, 'hidden_size') and hasattr(self.teacher.config, 'hidden_size') and self.model.config.hidden_size != self.teacher.config.hidden_size:
            self.proj = nn.Linear(self.model.config.hidden_size, self.teacher.config.hidden_size).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        student_outputs = model(**inputs, output_hidden_states=True)
        ce_loss = student_outputs.loss

        if self.distill_args.distillation_method == "none":
            return (ce_loss, student_outputs) if return_outputs else ce_loss

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs, output_hidden_states=True)
        
        mask = inputs["labels"] != IGNORE_INDEX
        s_logits_m, t_logits_m = student_outputs.logits[mask], teacher_outputs.logits[mask]
        
        distill_loss = torch.tensor(0.0, device=ce_loss.device)
        T, method = self.distill_args.temperature, self.distill_args.distillation_method
        
        if method == 'kl':
            distill_loss = nn.functional.kl_div(nn.functional.log_softmax(s_logits_m / T, -1), nn.functional.softmax(t_logits_m / T, -1), reduction='batchmean') * (T ** 2)
        elif method in ['sj', 'ssjs']:
            s_probs, t_probs = nn.functional.softmax(s_logits_m, -1), nn.functional.softmax(t_logits_m, -1)
            token_js = js_divergence(s_probs, t_probs).mean()
            if method == 'sj':
                distill_loss = token_js
            else: # ssjs
                s_full_probs = nn.functional.softmax(student_outputs.logits, -1)
                t_full_probs = nn.functional.softmax(teacher_outputs.logits, -1)
                s_trans = torch.einsum('bti,btj->btij', s_full_probs[:, :-1, :], s_full_probs[:, 1:, :])
                t_trans = torch.einsum('bti,btj->btij', t_full_probs[:, :-1, :], t_full_probs[:, 1:, :])
                trans_mask = (inputs["labels"][:, :-1] != IGNORE_INDEX) & (inputs["labels"][:, 1:] != IGNORE_INDEX)
                if trans_mask.any():
                    trans_js = js_divergence(s_trans[trans_mask], t_trans[trans_mask]).mean()
                else:
                    trans_js = torch.tensor(0.0, device=ce_loss.device)
                distill_loss = token_js + self.distill_args.lambda_sjs * trans_js
        
        s_hidden, t_hidden = student_outputs.hidden_states[-1], teacher_outputs.hidden_states[-1]
        if self.proj:
            s_hidden = self.proj(s_hidden)
        mse_loss = nn.functional.mse_loss(s_hidden[mask], t_hidden[mask])
        
        a, b = self.distill_args.alpha_ce, self.distill_args.beta_mse
        total_loss = a * ce_loss + (1 - a) * distill_loss + b * mse_loss
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                self.log({"peak_gpu_memory_mb": peak_mem})
        
        return (total_loss, student_outputs) if return_outputs else total_loss

class CrfTrainer(Trainer):
    """Custom trainer to handle CRF decoding during evaluation, especially for multi-GPU."""
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            emissions = outputs.logits
            mask = inputs["attention_mask"].bool()
            
            crf_model = model.module if hasattr(model, 'module') else model
            
            decoded_sequences = crf_model.crf.decode(emissions, mask=mask)

            max_len = inputs["labels"].shape[1]
            preds = torch.full(inputs["labels"].shape, IGNORE_INDEX, device=emissions.device, dtype=torch.long)
            for i, seq in enumerate(decoded_sequences):
                preds[i, :len(seq)] = torch.tensor(seq, device=emissions.device)
            
        return (loss, preds, inputs["labels"])

# =================================================================================
# 5. Metrics & Prediction Alignment
# =================================================================================

def align_predictions(predictions, label_ids, id2label, is_crf=False):
    # The 'predictions' object might be a tuple (logits, hidden_states). We only want the logits.
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    
    # For CRF, predictions are already decoded indices. Otherwise, argmax logits.
    preds = logits if is_crf else np.argmax(logits, axis=2)
    
    preds_list, labels_list = [], []
    for i in range(preds.shape[0]):
        pred_row, label_row = [], []
        for j in range(preds.shape[1]):
            if label_ids[i, j] != IGNORE_INDEX:
                pred_row.append(id2label.get(int(preds[i, j]), "O"))
                label_row.append(id2label.get(int(label_ids[i, j]), "O"))
        preds_list.append(pred_row)
        labels_list.append(label_row)
    return preds_list, labels_list

def build_compute_metrics(id2label, is_crf=False):
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        # p.predictions is what is returned by prediction_step
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds, golds = align_predictions(predictions, p.label_ids, id2label, is_crf)
        report = classification_report(golds, preds, output_dict=True, zero_division=0)
        # Use micro avg for overall token-level performance
        return {"f1": report["micro avg"]["f1-score"], "precision": report["micro avg"]["precision"], "recall": report["micro avg"]["recall"]}
    return compute_metrics

# =================================================================================
# 6. Main Benchmark Runner
# =================================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DistillArguments, TrainingArguments))
    m_args, d_args, dist_args, t_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
    set_seed(t_args.seed)

    labels = get_labels(d_args.labels)
    id2label = {i: l for i, l in enumerate(labels)}
    num_labels = len(labels)
    t_args.remove_unused_columns = False

    tokenizer_name = m_args.tokenizer_name or m_args.teacher_model_name_or_path or "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, add_prefix_space=True)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    teacher = None
    if m_args.model_type == 'distill' and dist_args.distillation_method != 'none':
        if not m_args.teacher_model_name_or_path:
            raise ValueError("A --teacher_model_name_or_path is required for distillation.")
        teacher_config = AutoConfig.from_pretrained(m_args.teacher_model_name_or_path, num_labels=num_labels, output_hidden_states=True)
        teacher = AutoModelForTokenClassification.from_pretrained(m_args.teacher_model_name_or_path, config=teacher_config)

    student_config_path = m_args.student_config_name
    if student_config_path and os.path.exists(student_config_path):
        student_config = StudentConfig.from_json_file(student_config_path)
    else:
        student_config = StudentConfig()
    student_config.num_labels = num_labels
    student_config.vocab_size = tokenizer.vocab_size

    student = None
    if m_args.model_type == 'crf':
        logger.info("Initializing Student model with CRF layer.")
        student = StudentModelWithCRF(student_config)
    elif m_args.model_type == 'distill':
        logger.info(f"Initializing Student model for {dist_args.distillation_method} distillation.")
        student = StudentModelForTokenClassification(student_config)
    else:
        raise ValueError("Invalid model_type. Choose 'distill' or 'crf'.")

    train_ds = NerDataset(d_args.data_dir, tokenizer, labels, d_args.max_seq_length, "train_dev", d_args.overwrite_cache) if t_args.do_train else None
    dev_ds = NerDataset(d_args.data_dir, tokenizer, labels, d_args.max_seq_length, "devel", d_args.overwrite_cache) if t_args.do_eval else None
    test_ds = NerDataset(d_args.data_dir, tokenizer, labels, d_args.max_seq_length, "test", d_args.overwrite_cache) if t_args.do_predict else None

    is_crf = m_args.model_type == 'crf'
    compute_metrics_fn = build_compute_metrics(id2label, is_crf=is_crf)

    trainer = None
    if is_crf:
        trainer = CrfTrainer(model=student, args=t_args, train_dataset=train_ds, eval_dataset=dev_ds, compute_metrics=compute_metrics_fn, data_collator=data_collator)
    else: # distill or vanilla
        trainer = DistillationTrainer(model=student, teacher_model=teacher, distill_args=dist_args, args=t_args, train_dataset=train_ds, eval_dataset=dev_ds, compute_metrics=compute_metrics_fn, data_collator=data_collator)

    if t_args.do_train:
        method_name = dist_args.distillation_method if m_args.model_type == 'distill' else 'CRF'
        logger.info(f"*** Starting Training for model: {m_args.model_type}, method: {method_name} ***")
        start_time = time.time()
        trainer.train(resume_from_checkpoint=t_args.resume_from_checkpoint)
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        trainer.save_model()
        trainer.log({"training_time_seconds": training_time})

    if t_args.do_eval:
        logger.info("*** Evaluating Student ***")
        metrics = trainer.evaluate()
        eval_time = metrics.get("eval_runtime", 0)
        logger.info(f"Student F1: {metrics.get('eval_f1', 0):.4f}")
        logger.info(f"Total evaluation time: {eval_time:.2f} seconds")

    if t_args.do_predict:
        if test_ds and len(test_ds) > 0:
            logger.info("*** Running Inference on Test Set ***")
            start_time = time.time()
            predict_results = trainer.predict(test_ds)
            end_time = time.time()
            inference_time = end_time - start_time
            logger.info(f"Total inference time on test set: {inference_time:.2f} seconds")
            
            preds_list, _ = align_predictions(predict_results.predictions, predict_results.label_ids, id2label, is_crf)
            output_predict_file = os.path.join(t_args.output_dir, "test_predictions.txt")
            with open(output_predict_file, "w") as writer:
                for sentence in preds_list:
                    writer.write(" ".join(sentence) + "\n")
        else:
            logger.warning("Test dataset not found or is empty. Skipping prediction.")

    if dev_ds and len(dev_ds) > 0:
        try:
            from fvcore.nn import FlopCountAnalysis
            logger.info("*** Analyzing Inference TFLOPs ***")
            batch = next(iter(DataLoader(dev_ds, batch_size=t_args.per_device_eval_batch_size, collate_fn=data_collator)))
            batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
            if "labels" in batch:
                batch.pop("labels")
            inputs_tuple = (batch['input_ids'], batch['attention_mask'])
            flops = FlopCountAnalysis(student, inputs_tuple)
            total_flops = flops.total()
            logger.info(f"Inference FLOPs per batch: {total_flops / 1e9:.4f} GFLOPs")
            if t_args.do_train:
                 trainer.log({"inference_gflops_per_batch": total_flops / 1e9})
        except ImportError:
            logger.warning("fvcore not installed. Skipping TFLOPs analysis. `pip install fvcore`")
        except Exception as e:
            logger.error(f"Could not run FLOPs analysis: {e}")

    logger.info("Benchmark run finished.")

if __name__ == "__main__":
    main()
