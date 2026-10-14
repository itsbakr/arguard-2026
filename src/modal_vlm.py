"""Modal app: LoRA fine-tune QCRI/MemeLens-VLM jointly for ArGuard A1 and A2.

Why this backbone. ``QCRI/MemeLens-VLM`` is a Qwen3-VL-8B-Instruct checkpoint the
ArGuard organizers' own lab fine-tuned on Arabic hateful-meme data, MIT licensed and
ungated, with ``ar`` in its language tags. It has already seen this taxonomy, which
base Qwen3-VL has not.

Why a classification head rather than generation. Our entire A2 edge is per-class
probability calibration: the decode ablation in ``src/a2_decode_ablation.py`` showed
+0.0697 macro-F1 from thresholding alone, and it needs clean, comparable per-class
scores. Generated text does not give those. So we pool the final hidden state and
attach one linear head emitting 11 logits: 1 binary for A1 and 10 sigmoid for A2.
One backbone, one training run, both subtasks.

Why the vision tower is frozen. 3,500 training images is far too few to move a vision
encoder without overfitting, and freezing it roughly halves the memory.

    modal run src/modal_vlm.py::smoke          # 50 steps, ~$3, proves the loop
    modal run src/modal_vlm.py::train          # full run
    modal run src/modal_vlm.py::predict --split dev_test
"""

from __future__ import annotations

import os

import modal

APP = modal.App("arguard-vlm")

MODEL_ID = "QCRI/MemeLens-VLM"
DATASET_ID = "QCRI/ArGuard-Task1"

A2_ACTIVE = (
    "Contempt", "Dehumanization", "Exclusion", "Humor", "Incitement",
    "Inferiority", "Mocking", "Other", "Sarcasm", "Slurs",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # torch 2.9.x has a Conv3d regression that makes the Qwen3-VL vision forward
        # 10-20x slower (falls back to aten::slow_conv_dilated3d instead of cuDNN):
        # github.com/pytorch/pytorch/issues/174051. 2.10 has the fix.
        "torch==2.10.0",
        # Qwen3-VL's processor imports AutoVideoProcessor, which hard-requires
        # torchvision even though we only ever feed it stills.
        "torchvision==0.25.0",
        os.environ.get("ARGUARD_TX", "transformers==4.57.1"),  # Qwen3-VL needs >=4.57
        "accelerate==1.10.1",
        "peft==0.17.1",
        "datasets==4.0.0",
        "pillow==11.3.0",
        "scikit-learn==1.7.2",
        "numpy<3",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

# One volume for weights + dataset cache + checkpoints, so the ~16GB model download
# happens once rather than on every run.
VOL = modal.Volume.from_name("arguard-cache", create_if_missing=True)
CACHE = "/cache"

# Modal secret exposing HF_TOKEN (modal secret create huggingface HF_TOKEN=...)
SECRET = modal.Secret.from_name(os.environ.get("ARGUARD_MODAL_SECRET", "huggingface"))

# H100/L40S/A100 all require a payment method on Modal. A10:2 gives 2 x 24GB = 48GB
# on the free tier, which fits an 8B bf16 LoRA once the model is sharded across both
# devices. Override with ARGUARD_GPU=H100 the moment a card is on the account: an H100
# is roughly 5x the throughput of pipeline-parallel A10s for this workload.
GPU = os.environ.get("ARGUARD_GPU", "A10:2")
TIMEOUT = 7200  # the 300s default silently kills training runs


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(lora_r: int = 32, lora_alpha: int = 64, attn: str = "sdpa"):
    """MemeLens-VLM backbone, frozen vision tower, LoRA on the LM, one 11-way head.

    ``device_map="auto"`` shards the model across whatever GPUs are present. On A10:2
    that is pipeline parallelism, so it buys memory rather than speed: expect roughly
    single-A10 throughput, not double.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation=attn
    )
    # If accelerate quietly put ANY module on cpu or disk, every forward pages
    # weights across the PCIe bus and an H100 runs like a laptop. Fail loudly.
    devmap = getattr(base, "hf_device_map", None)
    print(f"hf_device_map: {devmap}", flush=True)
    if devmap and any(str(v) in ("cpu", "disk") for v in devmap.values()):
        raise RuntimeError(f"model partially offloaded to CPU/disk: {devmap}")
    for name, param in base.named_parameters():
        if "visual" in name or "vision" in name:
            param.requires_grad = False

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base, lora)

    # The head has to live wherever the final hidden state comes out, which under
    # pipeline parallelism is the last shard, not cuda:0.
    hidden = base.config.text_config.hidden_size
    last_device = next(reversed(list(base.parameters()))).device
    head = torch.nn.Linear(hidden, 1 + len(A2_ACTIVE), dtype=torch.bfloat16).to(last_device)
    torch.nn.init.zeros_(head.bias)
    model.classifier = head
    return model


def pooled_logits(model, batch):
    """Last non-padding hidden state through the 11-way head.

    Right-padded batches mean the final real token differs per row, so index it
    explicitly rather than taking ``[:, -1]``.
    """
    import torch

    out = model.base_model.model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        output_hidden_states=True,
    )
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    mask = batch["attention_mask"].to(hidden.device)
    last_idx = mask.sum(dim=1) - 1
    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
    return model.classifier(pooled.to(model.classifier.weight.device))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
PROMPT = (
    "This is an Arabic meme. The text extracted from it by OCR is:\n"
    "{text}\n\n"
    "Judge whether the meme is hateful, and which fine-grained categories apply."
)


def make_collate(processor, max_pixels: int):
    """Build a collate_fn that renders each meme into the chat template.

    ``max_pixels`` caps visual tokens, which is the single biggest lever on both
    step time and memory for a Qwen-VL fine-tune.
    """
    import torch

    def collate(examples):
        texts, images = [], []
        for ex in examples:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(text=(ex["text"] or "")[:600])},
                ],
            }]
            texts.append(processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            img = ex["image"].convert("RGB")
            if img.width * img.height > max_pixels:
                scale = (max_pixels / (img.width * img.height)) ** 0.5
                img = img.resize((max(28, int(img.width * scale)),
                                  max(28, int(img.height * scale))))
            images.append(img)

        # Stay on CPU. This collate runs inside DataLoader worker processes, and
        # touching CUDA from a forked worker either crashes or silently serialises
        # everything onto the main process. The training loop moves the batch.
        batch = dict(processor(text=texts, images=images, return_tensors="pt",
                               padding=True, truncation=True, max_length=2048))

        if examples[0].get("label") is not None:
            batch["y_binary"] = torch.tensor(
                [[1.0 if ex["label"] == "Hateful" else 0.0] for ex in examples],
                dtype=torch.bfloat16)
            batch["y_multi"] = torch.tensor(
                [[1.0 if c in (ex["fine_grained_label"] or []) else 0.0 for c in A2_ACTIVE]
                 for ex in examples], dtype=torch.bfloat16)
        return batch

    return collate


def to_device(batch: dict, device: str = "cuda:0") -> dict:
    return {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@APP.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={CACHE: VOL},
              secrets=[SECRET], cpu=8, memory=32768,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def _train(max_steps: int, epochs: int, lr: float, batch_size: int,
           grad_accum: int, max_pixels: int, seed: int, tag: str,
           do_profile: bool = False) -> dict:
    import json
    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # keep_in_memory is load-bearing. HF_HOME sits on a Modal Volume (network FUSE),
    # and random-access parquet reads through it decompress a whole row group of
    # embedded images per item: measured ~4 min per optimizer step, 40+ hours per
    # run, with the H100 idle. The dataset is ~350MB; load it into RAM once.
    ds = load_dataset(DATASET_ID, keep_in_memory=True)
    if os.environ.get("ARGUARD_TRAIN_SPLITS", "train") == "train+dev":
        from datasets import concatenate_datasets
        train_ds = concatenate_datasets([ds["train"], ds["dev"]]).shuffle(seed=seed)
        print(f"training on train+dev: {len(train_ds)} rows", flush=True)
    else:
        train_ds = ds["train"].shuffle(seed=seed)
    collate = make_collate(processor, max_pixels)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=8, prefetch_factor=4,
                        persistent_workers=True, pin_memory=True)

    print(f"cuda device: {torch.cuda.get_device_name(0)}  "
          f"x{torch.cuda.device_count()}", flush=True)
    model = build_model()
    model.gradient_checkpointing_enable()
    # Without this, gradient checkpointing sees inputs with requires_grad=False and
    # silently drops gradients for every LoRA adapter inside a checkpointed block:
    # only the classifier head would train, on frozen features, and the loss still
    # falls so nothing looks wrong. torch warns "None of the inputs have
    # requires_grad=True" exactly once and is easy to miss.
    model.enable_input_require_grads()
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M", flush=True)

    # pos_weight from the train prior. Capped at 8 because Exclusion's 10/3500 would
    # otherwise produce a weight of 349 and destabilise the whole head.
    counts = np.array([[sum(1 for r in ds["train"]["fine_grained_label"] if c in (r or []))]
                       for c in A2_ACTIVE], dtype=float).ravel()
    n = len(train_ds)
    pos_weight = torch.tensor(
        np.clip(np.sqrt((n - counts) / np.maximum(counts, 1)), 1.0, 8.0),
        dtype=torch.bfloat16, device="cuda:0")
    n_hateful = sum(1 for lbl in ds["train"]["label"] if lbl == "Hateful")
    bin_weight = torch.tensor([(n - n_hateful) / max(n_hateful, 1)],
                              dtype=torch.bfloat16, device="cuda:0")

    # Under device_map="auto" the head, and therefore the logits, sit on the last
    # shard while the collated targets sit on cuda:0. Bind the loss lazily to
    # whatever device the logits actually come out on.
    _loss_cache: dict = {}

    def losses_for(device):
        if device not in _loss_cache:
            _loss_cache[device] = (
                torch.nn.BCEWithLogitsLoss(pos_weight=bin_weight.to(device)),
                torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)),
            )
        return _loss_cache[device]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = max_steps if max_steps > 0 else (len(loader) // grad_accum) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)

    import time
    t0 = time.time()
    step, losses = 0, []
    t_fetch = time.perf_counter()
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            # Phase timing for the first steps: separates dataloader wait from GPU
            # compute, so a throughput regression names its own culprit.
            t_have = time.perf_counter()
            batch = to_device(batch)
            logits = pooled_logits(model, batch)
            bce_bin, bce_multi = losses_for(logits.device)
            loss = (bce_bin(logits[:, :1], batch["y_binary"].to(logits.device))
                    + bce_multi(logits[:, 1:], batch["y_multi"].to(logits.device))) / grad_accum
            loss.backward()
            losses.append(float(loss) * grad_accum)
            if step < 6:
                torch.cuda.synchronize()
                t_done = time.perf_counter()
                seq = int(batch["input_ids"].shape[1])
                print(f"timing step~{step}: fetch {t_have - t_fetch:.2f}s  "
                      f"gpu {t_done - t_have:.2f}s  seq_len {seq}", flush=True)
                if i == 4 and do_profile:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        logits_p = pooled_logits(model, batch)
                        bb, bm = losses_for(logits_p.device)
                        lp = (bb(logits_p[:, :1], batch["y_binary"].to(logits_p.device))
                              + bm(logits_p[:, 1:], batch["y_multi"].to(logits_p.device)))
                        lp.backward()
                        torch.cuda.synchronize()
                    print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                                    row_limit=15), flush=True)
                    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                                    row_limit=10), flush=True)
            t_fetch = time.perf_counter()

            if (i + 1) % grad_accum == 0:
                if step == 0:
                    # Prove the LoRA adapters are actually receiving gradient, not
                    # just the head. Fail loudly rather than burn a training run.
                    lora_g = [p.grad.norm().item() for n, p in model.named_parameters()
                              if p.requires_grad and "lora" in n and p.grad is not None]
                    head_g = model.classifier.weight.grad
                    print(f"grad check: {len(lora_g)} lora tensors with grad, "
                          f"mean norm {np.mean(lora_g) if lora_g else 0:.3e}; "
                          f"head norm {head_g.norm().item() if head_g is not None else 0:.3e}",
                          flush=True)
                    if not lora_g or np.mean(lora_g) == 0:
                        raise RuntimeError(
                            "LoRA adapters received no gradient. Check that "
                            "enable_input_require_grads() runs before training."
                        )
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step <= 3 or step % 10 == 0:
                    rate = step / max(time.time() - t0, 1e-9)
                    print(f"epoch {epoch} step {step}/{total} "
                          f"loss {np.mean(losses[-10 * grad_accum:]):.4f} "
                          f"{rate * 60:.1f} steps/min "
                          f"eta {(total - step) / max(rate, 1e-9) / 60:.0f}m", flush=True)
                if 0 < max_steps <= step:
                    break
        if 0 < max_steps <= step:
            break
        ep_dir = f"{CACHE}/checkpoints/{tag}_ep{epoch}"
        model.save_pretrained(ep_dir)
        torch.save(model.classifier.state_dict(), f"{ep_dir}/classifier.pt")
        VOL.commit()
        print(f"checkpoint saved: {ep_dir}", flush=True)

    out = f"{CACHE}/checkpoints/{tag}"
    model.save_pretrained(out)
    torch.save(model.classifier.state_dict(), f"{out}/classifier.pt")
    VOL.commit()
    print(f"saved to {out}")
    return {"tag": tag, "steps": step, "final_loss": float(np.mean(losses[-50:]))}


@APP.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={CACHE: VOL},
              secrets=[SECRET], cpu=8, memory=32768)
def _predict(tag: str, split: str, max_pixels: int, batch_size: int) -> dict:
    """Score a split and return raw per-class probabilities.

    Thresholding happens locally, not here: the decode rule is fitted on
    out-of-fold scores and we want the raw probabilities to refit it against.
    """
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, keep_in_memory=True)[split]

    # Load the RAW base and attach only the trained adapter. Building via
    # build_model() first injects a fresh LoRA and then wraps the model a second
    # time ("Already found a peft_config" warning) - it happened to work because a
    # fresh adapter is zero-initialised, but which adapter is active becomes
    # ambiguous. One model, one adapter, no ambiguity.
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    ckpt = f"{CACHE}/checkpoints/{tag}"
    model = PeftModel.from_pretrained(base, ckpt, is_trainable=False)
    head = torch.nn.Linear(model.config.text_config.hidden_size, 1 + len(A2_ACTIVE),
                           dtype=torch.bfloat16).to("cuda")
    head.load_state_dict(torch.load(f"{ckpt}/classifier.pt"))
    model.classifier = head
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(processor, max_pixels), num_workers=8,
                        prefetch_factor=4, pin_memory=True)
    scores = []
    with torch.no_grad():
        for batch in loader:
            logits = pooled_logits(model, to_device(batch)).float()
            scores.append(torch.sigmoid(logits).cpu().numpy())

    import numpy as np
    arr = np.concatenate(scores, axis=0)
    return {
        "split": split,
        "ids": list(ds["id"]),
        "p_hateful": arr[:, 0].tolist(),
        "p_multi": arr[:, 1:].tolist(),
        "classes": list(A2_ACTIVE),
    }


@APP.function(image=image, gpu=GPU, timeout=1800, volumes={CACHE: VOL},
              secrets=[SECRET], cpu=8, memory=32768)
def _bisect() -> None:
    """Name the 7s-per-image culprit by elimination, not conjecture.

    Times, on the same batch: (0) a raw dense matmul as a silicon sanity check,
    (a) text-only forward+backward, (b) vision+text with checkpointing OFF,
    (c) vision+text with checkpointing ON. Prints one line each.
    """
    import time as _t

    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor

    def sync_time(fn, n=3):
        fn()  # warmup
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (_t.perf_counter() - t0) / n

    # (0) silicon check: 4096^3 bf16 matmul, expect >400 TFLOPs on a real H100
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    dt = sync_time(lambda: a @ a, n=10)
    print(f"bisect matmul: {2 * 4096**3 / dt / 1e12:.0f} TFLOPs", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, keep_in_memory=True)["train"]
    batch = make_collate(processor, 512 * 512)([ds[i] for i in range(2)])
    batch = to_device(batch)

    for attn in ("sdpa", "eager", "flash_attention_2"):
        try:
            model = build_model(attn=attn)
        except Exception as e:
            print(f"bisect attn={attn}: LOAD FAILED {str(e)[:120]}", flush=True)
            continue
        model.enable_input_require_grads()
        model.train()

        def fwd_bwd(use_pixels, model=model):
            b = dict(batch)
            if not use_pixels:
                b.pop("pixel_values", None)
                b.pop("image_grid_thw", None)
            logits = pooled_logits(model, b)
            logits.sum().backward()
            model.zero_grad(set_to_none=True)

        try:
            tv = sync_time(lambda: fwd_bwd(True))
            tt = sync_time(lambda: fwd_bwd(False))
            print(f"bisect attn={attn}: vision+text {tv:.2f}s | text-only {tt:.2f}s",
                  flush=True)
        except Exception as e:
            print(f"bisect attn={attn}: RUN FAILED {str(e)[:120]}", flush=True)
        del model
        torch.cuda.empty_cache()


@APP.local_entrypoint()
def bisect():
    _bisect.remote()


# --------------------------------------------------------------------------
# Entrypoints
# --------------------------------------------------------------------------
@APP.local_entrypoint()
def smoke():
    """50 steps on a tiny batch. Proves the image, the GPU, the loader and the loss."""
    print(_train.remote(max_steps=50, epochs=1, lr=1e-4, batch_size=1, grad_accum=8,
                        max_pixels=512 * 512, seed=42, tag="smoke", do_profile=True))


@APP.local_entrypoint()
def train(epochs: int = 3, lr: float = 1e-4, batch_size: int = 8, grad_accum: int = 2,
          max_pixels: int = 1024 * 1024, seed: int = 42, tag: str = "v1"):
    print(_train.remote(max_steps=0, epochs=epochs, lr=lr, batch_size=batch_size,
                        grad_accum=grad_accum, max_pixels=max_pixels, seed=seed, tag=tag))


@APP.local_entrypoint()
def predict(tag: str = "v1", split: str = "dev_test", max_pixels: int = 1024 * 1024,
            batch_size: int = 8):
    import json
    from pathlib import Path

    result = _predict.remote(tag=tag, split=split, max_pixels=max_pixels,
                             batch_size=batch_size)
    out = Path(__file__).resolve().parents[1] / "predictions" / f"vlm_{tag}_{split}.json"
    out.write_text(json.dumps(result))
    print(f"wrote {out}  ({len(result['ids'])} rows)")
