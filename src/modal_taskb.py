"""Modal app: fine-tune Arabic encoders for Task B, to ensemble with the char-ngram.

The char-ngram baseline already sits at B1 0.9203 / B2 0.8620 (leaderboard-verified to
seven decimals), so encoders only need to add diversity, not carry the system.
Each run is a bert-base on 16.8k short prompts: minutes on an A10, so we fan out
(model x seed) pairs concurrently and average calibrated probabilities locally.

One backbone, two heads (binary safe/unsafe + 7-way harm domain over ALL rows),
multi-task loss, dialect prepended as a literal token -- mirroring the char-ngram
setup so ensemble members disagree on representation, not on task framing.

    modal run src/modal_taskb.py::sweep
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP = modal.App("arguard-taskb")

LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "taskB"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.9.1",
        "transformers==4.57.1",
        "datasets==4.0.0",
        "scikit-learn==1.7.2",
        "accelerate==1.10.1",
        "numpy<3",
    )
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(LOCAL_DATA, "/data")
)

VOL = modal.Volume.from_name("arguard-cache", create_if_missing=True)
# Modal secret exposing HF_TOKEN (modal secret create huggingface HF_TOKEN=...)
SECRET = modal.Secret.from_name(os.environ.get("ARGUARD_MODAL_SECRET", "huggingface"))

B2_LABELS = ("adult_content", "harm-to-others", "self-harm", "harassment",
             "FraudandDeception", "Bully", "hate_speech")

MODELS = (
    "UBC-NLP/MARBERTv2",
    "aubmindlab/bert-base-arabertv02-twitter",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix",
)
SEEDS = (42, 43, 44)


@APP.function(image=image, gpu="A10", timeout=3600, volumes={"/cache": VOL},
              secrets=[SECRET], cpu=4, memory=16384)
def train_one(model_id: str, seed: int, epochs: int = 3, lr: float = 2e-5,
              batch_size: int = 32, max_len: int = 96, balance: bool = False) -> dict:
    """Fine-tune one encoder; return dev probabilities for local ensembling."""
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModel, AutoTokenizer,
                              get_cosine_schedule_with_warmup)

    torch.manual_seed(seed)
    np.random.seed(seed)

    def read(name, labelled=True):
        with open(f"/data/{name}", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if labelled:
            rows = [r for r in rows if r.get("label") and r.get("granular_label")]
        return rows

    train_rows = read("train.csv") + read("dev_with_label.csv")
    if balance:
        # The B1 test is ~49% unsafe vs 80% in train. Oversample safe rows to parity
        # so the decision boundary is learned for the world it will be scored in.
        import random as _rnd
        _rnd.seed(seed)
        safe = [r for r in train_rows if r["label"] == "safe"]
        unsafe = [r for r in train_rows if r["label"] == "unsafe"]
        train_rows = unsafe + safe * (len(unsafe) // len(safe)) + _rnd.sample(safe, len(unsafe) % len(safe))
        print(f"balanced training: {len(train_rows)} rows "
              f"({sum(r['label']=='unsafe' for r in train_rows)} unsafe)", flush=True)
    test_b1 = read("Arguard_binary_test_no_label.csv", labelled=False)
    test_b2 = read("Arguard_granular_test_no_label.csv", labelled=False)

    def prep(r):
        return f"[{r.get('dialect') or 'UNK'}] {r['prompt']}"

    tok = AutoTokenizer.from_pretrained(model_id)

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            item = tok(prep(r), truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in item.items()}
            item["y_bin"] = torch.tensor(1.0 if r["label"] == "unsafe" else 0.0)
            item["y_gran"] = torch.tensor(B2_LABELS.index(r["granular_label"]))
            return item

    backbone = AutoModel.from_pretrained(model_id).cuda()
    hidden = backbone.config.hidden_size
    head_bin = torch.nn.Linear(hidden, 1).cuda()
    head_gran = torch.nn.Linear(hidden, len(B2_LABELS)).cuda()

    params = (list(backbone.parameters()) + list(head_bin.parameters())
              + list(head_gran.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loader = DataLoader(DS(train_rows), batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    total = len(loader) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * total), total)
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()

    backbone.train()
    step = 0
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            out = backbone(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            pooled = out.last_hidden_state[:, 0]
            loss = (bce(head_bin(pooled).squeeze(-1), batch["y_bin"])
                    + ce(head_gran(pooled), batch["y_gran"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 100 == 0:
                print(f"{model_id} seed={seed} step {step}/{total} "
                      f"loss {float(loss):.4f}", flush=True)

    backbone.eval()

    class UDS(Dataset):
        """Unlabelled rows: text only."""

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            item = tok(prep(self.rows[i]), truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in item.items()}

    def infer(rows):
        loader = DataLoader(UDS(rows), batch_size=128, num_workers=4)
        pu, pg = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                pooled = backbone(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"]).last_hidden_state[:, 0]
                pu.extend(torch.sigmoid(head_bin(pooled)).squeeze(-1).float().cpu().tolist())
                pg.extend(torch.softmax(head_gran(pooled), -1).float().cpu().tolist())
        return pu, pg

    pu1, _ = infer(test_b1)
    _, pg2 = infer(test_b2)
    return {
        "model": model_id, "seed": seed, "classes": list(B2_LABELS),
        "b1_ids": [r["id"] for r in test_b1], "b1_p_unsafe": pu1,
        "b2_ids": [r["id"] for r in test_b2], "b2_p_gran": pg2,
    }


@APP.local_entrypoint()
def sweep_balanced():
    import json

    out_dir = Path(__file__).resolve().parents[1] / "predictions" / "taskb_test_bal"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, 3, 2e-5, 32, 96, True) for m in MODELS for s in SEEDS]
    print(f"fanning out {len(jobs)} BALANCED runs on A10s")
    for result in train_one.starmap(jobs):
        name = result["model"].split("/")[-1]
        path = out_dir / f"{name}_seed{result['seed']}.json"
        path.write_text(json.dumps(result))
        print(f"done: {name} seed {result['seed']}", flush=True)


@APP.local_entrypoint()
def sweep():
    import json

    out_dir = Path(__file__).resolve().parents[1] / "predictions" / "taskb_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in MODELS for s in SEEDS]
    print(f"fanning out {len(jobs)} runs on A10s")
    for result in train_one.starmap(jobs):
        name = result["model"].split("/")[-1]
        path = out_dir / f"{name}_seed{result['seed']}.json"
        path.write_text(json.dumps(result))
        print(f"done: {name} seed {result['seed']} -> {path.name}")
