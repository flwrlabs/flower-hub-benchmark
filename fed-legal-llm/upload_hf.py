#!/usr/bin/env python3
"""
Build an English-only 5-silo legal federated SFT dataset and optionally push it to Hugging Face.

Chosen silos (all supervised / self-contained):
  0: LexGLUE / LEDGAR                                  -> legal provision classification
  1: LexGLUE / CaseHOLD                                -> multiple-choice case holding selection
  2: LexGLUE / Unfair_ToS                              -> consumer ToS classification
  3: LexGLUE / SCOTUS                                  -> Supreme Court issue classification
  4: LegalBench / merged contract_nli_* configurations -> contract NLI / legal reasoning

Output:
  DatasetDict with splits:
    - train
    - valid
    - test

Each row includes:
  - messages   : chat-style messages list
  - client_id  : int in {0,1,2,3,4}
  - task_type  : classification | multiple_choice | multi_label_classification
  - language   : "en"
  - example_id : deterministic hash/id when possible

Per-client splitting:
  - 5% valid
  - 5% test
  - 90% train

Global valid/test are formed by concatenating each client's held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    disable_caching,
    load_dataset,
)

disable_caching()

SYSTEM_PROMPT = (
    "You are a legal assistant. Answer concisely and accurately. "
    "When legal text is provided, rely on it and do not invent facts."
)

CLIENT_SPECS = [
    {
        "client_id": 0,
        "dataset_name": "coastalcph/lex_glue",
        "config": "ledgar",
        "builder": "build_ledgar",
        "task_type": "classification",
        "sample_fraction": 0.5,  # use half the data
    },
    {
        "client_id": 1,
        "dataset_name": "coastalcph/lex_glue",
        "config": "case_hold",
        "builder": "build_case_hold",
        "task_type": "multiple_choice",
        "sample_fraction": 0.5,  # use half the data
    },
    {
        "client_id": 2,
        "dataset_name": "coastalcph/lex_glue",
        "config": "unfair_tos",
        "builder": "build_unfair_tos",
        "task_type": "multi_label_classification",
        "sample_fraction": 1.0,
    },
    {
        "client_id": 3,
        "dataset_name": "coastalcph/lex_glue",
        "config": "scotus",
        "builder": "build_scotus",
        "task_type": "classification",
        "sample_fraction": 1.0,
    },
    {
        "client_id": 4,
        "dataset_name": "nguha/legalbench",
        "config": [
            "contract_nli_confidentiality_of_agreement",
            "contract_nli_limited_use",
            "contract_nli_no_licensing",
            "contract_nli_notice_on_compelled_disclosure",
            "contract_nli_sharing_with_employees",
            "contract_nli_sharing_with_third-parties",
            "contract_nli_survival_of_obligations",
        ],
        "builder": "build_legalbench_contract_nli",
        "task_type": "classification",
        "sample_fraction": 1.0,
    },
]


# ----------------------------
# generic helpers
# ----------------------------

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_ws(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_or_empty(example: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, list):
                value = " ".join(map(str, value))
            return normalize_ws(value)
    return ""


def stringify_label(value: Any, feature: Any = None) -> str:
    if isinstance(feature, ClassLabel):
        try:
            return str(feature.int2str(int(value)))
        except Exception:
            pass
    return normalize_ws(value)


def make_messages(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": normalize_ws(user)},
        {"role": "assistant", "content": normalize_ws(assistant)},
    ]


def finalize_record(
    messages: List[Dict[str, str]],
    client_id: int,
    task_type: str,
    example_id: Optional[str] = None,
) -> Dict[str, Any]:
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return {
        "messages": messages,
        "client_id": int(client_id),
        "task_type": task_type,
        "language": "en",
        "example_id": example_id or sha1_text(canonical),
    }


def deterministic_shuffle(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = list(records)
    rng.shuffle(out)
    return out


def maybe_downsample(
    records: List[Dict[str, Any]],
    seed: int,
    sample_fraction: float = 1.0,
) -> List[Dict[str, Any]]:
    if sample_fraction >= 1.0:
        return records
    if sample_fraction <= 0.0:
        raise ValueError(f"sample_fraction must be > 0, got {sample_fraction}")

    n = len(records)
    target_n = max(1, int(round(n * sample_fraction)))
    records = deterministic_shuffle(records, seed)
    return records[:target_n]


def split_records(
    records: List[Dict[str, Any]],
    seed: int,
    test_frac: float = 0.05,
    valid_frac: float = 0.05,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Simple deterministic split per client.
    We do not stratify because labels differ across tasks and some rows are multi-label / free-form QA.
    """
    n = len(records)
    if n < 20:
        raise ValueError(f"Client has only {n} examples; too small for 5%/5% split.")
    records = deterministic_shuffle(records, seed)
    n_test = max(1, int(round(n * test_frac)))
    n_valid = max(1, int(round(n * valid_frac)))
    if n_test + n_valid >= n:
        raise ValueError(f"Split would exhaust dataset: n={n}, test={n_test}, valid={n_valid}")
    test = records[:n_test]
    valid = records[n_test:n_test + n_valid]
    train = records[n_test + n_valid:]
    return {"train": train, "valid": valid, "test": test}


def to_dataset(records: List[Dict[str, Any]]) -> Dataset:
    if not records:
        raise ValueError("No records to convert into Dataset.")
    return Dataset.from_list(records)


# ----------------------------
# dataset-specific builders
# ----------------------------

def build_ledgar(ds: Dataset, client_id: int, task_type: str) -> List[Dict[str, Any]]:
    out = []
    feat = ds.features.get("label")

    for ex in ds:
        text = text_or_empty(ex, "text", "context", "sentence", "clause", "premise")
        if not text:
            continue

        raw_labels = None
        for key in ("labels", "label"):
            if key in ex:
                raw_labels = ex[key]
                break

        if raw_labels is None:
            continue

        if isinstance(raw_labels, list):
            labels = [stringify_label(v, feat if not isinstance(feat, dict) else None) for v in raw_labels]
        else:
            labels = [stringify_label(raw_labels, feat)]

        labels = [normalize_ws(x) for x in labels if normalize_ws(x)]
        if not labels:
            continue

        user = (
            "Read the legal provision and list the applicable contract clause categories.\n\n"
            f"Provision:\n{text}"
        )
        assistant = ", ".join(labels)
        ex_id = ex.get("id") or sha1_text(text + assistant)
        out.append(finalize_record(make_messages(user, assistant), client_id, task_type, ex_id))
    return out


def build_case_hold(ds: Dataset, client_id: int, task_type: str) -> List[Dict[str, Any]]:
    out = []
    label_feature = ds.features.get("label")

    for ex in ds:
        context = text_or_empty(ex, "context", "text", "premise")
        if not context:
            continue

        endings = None
        for k in ("endings", "choices", "options"):
            if k in ex and isinstance(ex[k], list):
                endings = ex[k]
                break
        if endings is None:
            endings = [ex[k] for k in sorted(ex.keys()) if re.fullmatch(r"ending_\d+", k)]
        if not endings:
            continue

        label_val = ex.get("label")
        if label_val is None:
            continue
        try:
            idx = int(label_val)
        except Exception:
            idx = None

        option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = []
        for i, option in enumerate(endings):
            option_lines.append(f"{option_letters[i]}. {normalize_ws(option)}")

        if idx is not None and 0 <= idx < len(endings):
            assistant = f"The best answer is {option_letters[idx]}. {normalize_ws(endings[idx])}"
        else:
            assistant = stringify_label(label_val, label_feature)

        user = (
            "Read the case excerpt and choose the best holding from the options.\n\n"
            f"Case excerpt:\n{context}\n\n"
            "Options:\n" + "\n".join(option_lines)
        )
        ex_id = ex.get("id") or sha1_text(context + assistant)
        out.append(finalize_record(make_messages(user, assistant), client_id, task_type, ex_id))
    return out


def build_unfair_tos(ds: Dataset, client_id: int, task_type: str) -> List[Dict[str, Any]]:
    out = []

    label_feature = None
    if "labels" in ds.features:
        label_feature = ds.features["labels"]
    elif "label" in ds.features:
        label_feature = ds.features["label"]

    def decode_multilabel(raw_value):
        if isinstance(raw_value, list):
            vals = raw_value
        else:
            vals = [raw_value]

        decoded = []
        for v in vals:
            if v is None:
                continue

            if isinstance(v, str) and not v.strip():
                continue

            names = getattr(getattr(label_feature, "feature", None), "names", None)
            if names is not None:
                try:
                    vi = int(v)
                    if 0 <= vi < len(names):
                        decoded.append(str(names[vi]))
                        continue
                except Exception:
                    pass

            if isinstance(label_feature, ClassLabel):
                try:
                    decoded.append(str(label_feature.int2str(int(v))))
                    continue
                except Exception:
                    pass

            decoded.append(normalize_ws(v))

        decoded = [x for x in decoded if x]

        seen = set()
        result = []
        for x in decoded:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    for ex in ds:
        text = text_or_empty(ex, "text", "sentence", "context")
        if not text:
            continue

        raw_labels = ex.get("labels", ex.get("label", None))
        if raw_labels is None:
            continue

        labels = decode_multilabel(raw_labels)
        if not labels:
            labels = ["None"]

        user = (
            "Read the following Terms of Service clause and list all applicable unfairness categories. "
            "If none apply, answer with None.\n\n"
            f"Clause:\n{text}"
        )
        assistant = ", ".join(labels)

        ex_id = ex.get("id") or sha1_text(text + assistant)
        out.append(finalize_record(make_messages(user, assistant), client_id, task_type, ex_id))

    return out


def build_scotus(ds: Dataset, client_id: int, task_type: str) -> List[Dict[str, Any]]:
    out = []
    label_feature = ds.features.get("label")

    for ex in ds:
        text = text_or_empty(ex, "text", "context", "sentence")
        if not text:
            continue

        if "label" not in ex or ex["label"] is None:
            continue

        label = stringify_label(ex["label"], label_feature)
        if not label:
            continue

        user = (
            "Read the following U.S. Supreme Court case excerpt and classify it into the correct legal issue area.\n\n"
            f"Excerpt:\n{text}"
        )
        assistant = label

        ex_id = ex.get("id") or sha1_text(text + label)
        out.append(finalize_record(make_messages(user, assistant), client_id, task_type, ex_id))

    return out


def build_legalbench_contract_nli(ds: Dataset, client_id: int, task_type: str) -> List[Dict[str, Any]]:
    out = []
    label_feature = ds.features.get("label")

    for ex in ds:
        prompt = text_or_empty(ex, "prompt", "question", "input", "text")
        answer = text_or_empty(ex, "answer", "target", "output")

        if prompt and answer:
            user = prompt
            assistant = answer
        else:
            premise = text_or_empty(ex, "premise", "contract", "context", "document")
            hypothesis = text_or_empty(ex, "hypothesis", "statement", "query")
            raw_label = ex.get("label")

            if not premise or not hypothesis or raw_label is None:
                continue

            label = stringify_label(raw_label, label_feature)
            label_norm = label.strip().lower()

            if label_norm in {"yes", "true", "entailment", "supported"}:
                assistant = "Entailment"
            elif label_norm in {"no", "false", "contradiction", "unsupported"}:
                assistant = "Contradiction"
            elif label_norm in {"neutral", "unknown", "not enough information", "neither"}:
                assistant = "Not enough information"
            else:
                assistant = label

            user = (
                "Read the contract text and determine whether it supports the statement.\n\n"
                f"Contract:\n{premise}\n\n"
                f"Statement:\n{hypothesis}\n\n"
                "Answer with one of: Entailment, Contradiction, Not enough information."
            )

        ex_id = ex.get("id") or sha1_text(user[:500] + assistant)
        out.append(finalize_record(make_messages(user, assistant), client_id, task_type, ex_id))

    return out


# ----------------------------
# loading / combining
# ----------------------------

def merge_hf_splits(ds_dict: DatasetDict) -> Dataset:
    parts = []
    for split_name in ds_dict.keys():
        parts.append(ds_dict[split_name])
    if not parts:
        raise ValueError("No splits found in dataset.")
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def load_raw_dataset(name: str, config):
    if isinstance(config, list):
        parts = []
        for cfg in config:
            ds = load_dataset(name, cfg)
            if isinstance(ds, DatasetDict):
                ds = merge_hf_splits(ds)
            parts.append(ds)
        return concatenate_datasets(parts)

    if config:
        ds = load_dataset(name, config)
    else:
        ds = load_dataset(name)

    if isinstance(ds, DatasetDict):
        return merge_hf_splits(ds)
    return ds


def build_client_records(spec: Dict[str, Any], seed: int, max_examples: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    raw = load_raw_dataset(spec["dataset_name"], spec["config"])
    builder_fn = globals()[spec["builder"]]
    records = builder_fn(
        raw,
        client_id=spec["client_id"],
        task_type=spec["task_type"],
    )

    dedup = {}
    for rec in records:
        dedup[rec["example_id"]] = rec
    records = list(dedup.values())

    sample_fraction = float(spec.get("sample_fraction", 1.0))
    records = maybe_downsample(records, seed=seed, sample_fraction=sample_fraction)

    if max_examples is not None and len(records) > max_examples:
        records = deterministic_shuffle(records, seed)[:max_examples]

    return split_records(records, seed=seed, test_frac=0.05, valid_frac=0.05)


def build_dataset(seed: int = 42, max_examples_per_client: Optional[int] = None) -> DatasetDict:
    global_train = []
    global_valid = []
    global_test = []

    for i, spec in enumerate(CLIENT_SPECS):
        client_seed = seed + 1000 * (i + 1)
        splits = build_client_records(spec, seed=client_seed, max_examples=max_examples_per_client)
        global_train.extend(splits["train"])
        global_valid.extend(splits["valid"])
        global_test.extend(splits["test"])

        cfg = spec["config"]
        cfg_name = ",".join(cfg) if isinstance(cfg, list) else str(cfg)
        print(
            f"client_{spec['client_id']} | {spec['dataset_name']}/{cfg_name:<40} "
            f"sample_fraction={spec.get('sample_fraction', 1.0):.2f} "
            f"train={len(splits['train']):>6} valid={len(splits['valid']):>5} test={len(splits['test']):>5}"
        )

    train_ds = to_dataset(deterministic_shuffle(global_train, seed + 11))
    valid_ds = to_dataset(deterministic_shuffle(global_valid, seed + 22))
    test_ds = to_dataset(deterministic_shuffle(global_test, seed + 33))
    return DatasetDict({"train": train_ds, "valid": valid_ds, "test": test_ds})


# ----------------------------
# hub / save
# ----------------------------

def maybe_login_hf(token: Optional[str]) -> None:
    if not token:
        return
    from huggingface_hub import login
    login(token=token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./legal_federated_5silo_en")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="flwrlabs/fed-legal",
        help="HF repo id, e.g. username/legal-federated-5silo-en",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples_per_client", type=int, default=None)
    parser.add_argument("--save_jsonl", action="store_true", help="Also save train/valid/test JSONL files")
    args = parser.parse_args()

    ds = build_dataset(seed=args.seed, max_examples_per_client=args.max_examples_per_client)

    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"\nSaved DatasetDict to: {args.output_dir}")

    if args.save_jsonl:
        for split in ("train", "valid", "test"):
            path = os.path.join(args.output_dir, f"{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in ds[split]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Saved {split} JSONL to: {path}")

    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# legal-federated-5silo-en\n\n"
            "English-only 5-silo legal federated SFT dataset.\n\n"
            "## Silos\n"
            "- 0: LexGLUE / LEDGAR (50% sampled)\n"
            "- 1: LexGLUE / CaseHOLD (50% sampled)\n"
            "- 2: LexGLUE / Unfair_ToS\n"
            "- 3: LexGLUE / SCOTUS\n"
            "- 4: LegalBench / merged contract_nli_* configs\n\n"
            "## Row schema\n"
            "- messages\n"
            "- client_id (int in {0,1,2,3,4})\n"
            "- task_type\n"
            "- language\n"
            "- example_id\n\n"
            "## Splits\n"
            "- train: 90% per client\n"
            "- valid: 5% per client\n"
            "- test: 5% per client\n\n"
            "Global valid/test are concatenations of each client's held-out split.\n"
        )
    print(f"Saved dataset card stub to: {readme_path}")

    if args.repo_id:
        maybe_login_hf(args.token)
        ds.push_to_hub(args.repo_id, private=args.private)
        print(f"Pushed dataset to hub: {args.repo_id}")
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=args.token)
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print("Uploaded README.md to the Hub.")
        except Exception as e:
            print(f"Warning: dataset push succeeded or partially succeeded, but README upload failed: {e}")


if __name__ == "__main__":
    main()
    