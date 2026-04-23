"""fed_legal_llm: Federated LoRA fine-tuning for legal instruction tuning."""

import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

HF_DATASET_ID = "flwrlabs/fed-legal"
_fds: FederatedDataset | None = None


def _get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SFTCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        rendered = []
        for ex in batch:
            messages = ex["messages"]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            rendered.append(text)

        enc = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def macro_mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def _make_partitioner(partitioner_name: str, num_partitions: int):
    name = partitioner_name.strip().lower()
    if name == "iid":
        return IidPartitioner(num_partitions=num_partitions)
    if name == "natural":
        return NaturalIdPartitioner(partition_by="client_id")
    raise ValueError(f"Unsupported partitioner '{partitioner_name}'. Use 'iid' or 'natural'.")


class LegalLoraModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: List[str],
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {}
        if torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        base_model.config.use_cache = False
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        self.model = get_peft_model(base_model, peft_cfg)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in state.items() if "lora_" in k}

    def set_lora_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        current = self.model.state_dict()
        for k, v in state_dict.items():
            if k in current:
                current[k] = v.to(current[k].dtype)
        self.model.load_state_dict(current, strict=strict)


def load_model_from_config(context) -> LegalLoraModel:
    target_modules = [m.strip() for m in str(context.run_config["lora-target-modules"]).split(",") if m.strip()]
    return LegalLoraModel(
        model_name=str(context.run_config["model-name"]),
        lora_r=int(context.run_config["lora-r"]),
        lora_alpha=int(context.run_config["lora-alpha"]),
        lora_dropout=float(context.run_config["lora-dropout"]),
        target_modules=target_modules,
        load_in_4bit=bool(context.run_config.get("load-in-4bit", False)),
        torch_dtype=str(context.run_config.get("torch-dtype", "auto")),
    )


def _load_dataset_source(dataset_id_or_path: str):
    if Path(dataset_id_or_path).exists():
        obj = load_from_disk(dataset_id_or_path)
        return obj
    return load_dataset(dataset_id_or_path)


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    tokenizer,
    max_length: int,
    partitioner_name: str = "natural",
    dataset_id: str = HF_DATASET_ID,
) -> DataLoader:
    global _fds
    if _fds is None:
        partitioner = _make_partitioner(partitioner_name, num_partitions)
        _fds = FederatedDataset(dataset=dataset_id, partitioners={"train": partitioner})
    partition = _fds.load_partition(partition_id) #.select(range(100))
    collator = SFTCollator(tokenizer, max_length=max_length)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, collate_fn=collator)


def load_local_data(data_path: str, batch_size: int, tokenizer, max_length: int) -> DataLoader:
    ds = load_from_disk(data_path)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    collator = SFTCollator(tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)


def load_centralized_testset(dataset_id: str = HF_DATASET_ID):
    obj = _load_dataset_source(dataset_id)
    if isinstance(obj, DatasetDict):
        return obj["test"] #.select(range(100))
    return load_dataset(dataset_id, split="test") #.select(range(100))


def train(model: LegalLoraModel, trainloader: DataLoader, epochs: int, lr: float, weight_decay: float, device: torch.device, show_progress: bool = False) -> float:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    running_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        progress = tqdm(trainloader, desc=f"Train {epoch + 1}/{epochs}", leave=False, disable=not show_progress)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())
            running_loss += loss_val
            steps += 1
            progress.set_postfix(loss=f"{loss_val:.4f}")

    return running_loss / max(steps, 1)


@torch.no_grad()
def evaluate_generation(
    model: LegalLoraModel,
    dataset: HFDataset,
    device: torch.device,
    max_new_tokens: int = 64,
    eval_batch_size: int = 4,
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    per_client: Dict[int, Dict[str, List[float]]] = {}
    all_f1: List[float] = []
    all_em: List[float] = []

    for start in tqdm(range(0, len(dataset), eval_batch_size), desc="Server eval", leave=False):
        batch = dataset.select(range(start, min(start + eval_batch_size, len(dataset))))
        prompts = []
        golds = []
        client_ids = []
        for ex in batch:
            msgs = ex["messages"]
            prompt_msgs = msgs[:-1]
            prompts.append(
                tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            )
            golds.append(msgs[-1]["content"])
            client_ids.append(int(ex["client_id"]))

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(getattr(tokenizer, "model_max_length", 2048), 2048),
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_only = generated[:, enc["input_ids"].shape[1]:]
        preds = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for pred, gold, cid in zip(preds, golds, client_ids):
            f1 = token_f1(pred, gold)
            em = exact_match(pred, gold)
            all_f1.append(f1)
            all_em.append(em)
            if cid not in per_client:
                per_client[cid] = {"f1": [], "em": []}
            per_client[cid]["f1"].append(f1)
            per_client[cid]["em"].append(em)

    metrics: Dict[str, float] = {
        "test_token_f1": float(macro_mean(all_f1)),
        "test_exact_match": float(macro_mean(all_em)),
    }

    client_macro_f1 = []
    client_macro_em = []
    for cid in sorted(per_client):
        f1 = float(macro_mean(per_client[cid]["f1"]))
        em = float(macro_mean(per_client[cid]["em"]))
        metrics[f"client_{cid}_token_f1"] = f1
        metrics[f"client_{cid}_exact_match"] = em
        client_macro_f1.append(f1)
        client_macro_em.append(em)

    metrics["macro_client_token_f1"] = float(macro_mean(client_macro_f1))
    metrics["macro_client_exact_match"] = float(macro_mean(client_macro_em))
    return metrics


def cosine_annealing(current_round: int, total_round: int, lrate_max: float = 5e-5, lrate_min: float = 1e-5) -> float:
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
