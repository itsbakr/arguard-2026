# Ahmed Younis at ArGuard Shared Tasks (ArabicNLP 2026)

Code for the system-description paper *"Calibration Is Most of What Is Left to
Tune: Decode-Time Structure and Threshold Transport for Arabic Harmful-Content
Detection"* (ArabicNLP 2026 Shared Tasks track, co-located with EMNLP 2026,
Budapest). Team **Ahmed Younis**, Codabench username `ahmedbakr`.

Official results: **1st** on subtask A2 (fine-grained hateful memes, macro-F1
0.4194), **3rd** on A1 (binary memes, 0.7889), **3rd** on B2 (harm domain,
0.7695), 7th on B1 (safe/unsafe, 0.7552).

**Paper:** published in the ArabicNLP 2026 shared task proceedings as
[2026.arabicnlp-sharedtasks.61](https://aclanthology.org/2026.arabicnlp-sharedtasks.61/)
(the 10 Sep 2026 camera-ready). A corrected version is in
[`paper/Younis_ArGuard2026_corrected.pdf`](paper/Younis_ArGuard2026_corrected.pdf);
I'm filing it as an ACL Anthology revision. See
[Paper corrections](#paper-corrections) for what changed.

## Contents

```
src/                   # one Python package; every script imports it as src.*
  taska_baseline.py    # Task A char-ngram member, label sets, gated decode with floors
  a2_decode_ablation.py  # A2 decode ablation, rules A to F (Table 1)
  modal_vlm.py         # Task A VLM: LoRA fine-tune and prediction on Modal
  vlm_decode.py        # decode rule F for one VLM run
  vlm_ensemble.py      # seed averaging, soft A1 gate, rank decode
  taskb_baseline.py    # Task B char-ngram member and data loader
  modal_taskb.py       # Task B: nine encoders on Modal (sweep, sweep_balanced)
  taskb_ensemble.py    # Task B dev-phase ensemble
  taskb_test_submit.py # Task B test build; produced the final B2 file
  build_submissions.py # packages the A1/A2 zips (and the dev-phase B1/B2 zips)
  submit.py            # builds and validates all four Codabench submission
                       # formats (the Terms tab, not the repo README, is
                       # authoritative for Task B), using zipfile so macOS
                       # metadata never breaks the archive
  scorer_b.py          # offline Task B scorer (written, never run)
run_vlm_pipeline.sh    # Task A dev phase: predict, decode rule F, package, check
taskA_memes/           # Task A decode scripts (dev_test and test phase)
taskB_prompts/         # Task B dev predictions and the B1 threshold builds
  probes/              # B1/B2 diagnostics and leaderboard probes
history/               # earlier module states behind some dev-phase numbers
audit/                 # the original 29-30 Jul data audits: id leak, near-duplicates, AraSafe overlap
id_leak_audit.py       # reproduces the Task B sequential-id label leak from the
                       # public dev-phase CSVs alone: contiguous per-label id runs,
                       # a nearest-id lookup fitted on train scoring 1.0000 dev
                       # macro-F1 with zero characters of text read, and the
                       # shuffled-id randomization control that collapses it to
                       # chance. Identified 29 Jul 2026 and left unexploited; the
                       # official test release is re-indexed from 1, so the artifact
                       # does not affect evaluation-phase scores.
```

## Pipeline

Paper section or result, and the code behind it. Commands run from the
repository root.

| Paper | Code |
|---|---|
| S2: dialect token, empty-label rows dropped | `src/taskb_baseline.py` (`load`, `_prep`); `src/modal_taskb.py` (`prep`) |
| S2: offline dev scores matched the dev leaderboard | `python -m src.taskb_baseline --fit --report` (B1 0.9203, B2 0.8620). `src/scorer_b.py` was written for this but never run. |
| S3: Task A VLM, LoRA, 11-logit head, weighted BCE | `src/modal_vlm.py` (`train`, `predict`) |
| S3: Task B nine encoders, two heads | `src/modal_taskb.py::sweep` (test phase, train+dev); `history/taskB_devphase/modal_taskb.py` (dev phase, train only) |
| S3: B2 blend with the char-ngram, argmax | `src/taskb_test_submit.py` |
| S4.1, Table 1: A2 decode rules A to F | `python -m src.a2_decode_ablation` (0.2559 to 0.3256, char-ngram scores) |
| S4.2: rank decoding against known counts | `src/vlm_ensemble.py` (dev_test); `taskA_memes/` (test) |
| S4.3: first B1 test file, 0.5959 | `src/taskb_test_submit.py`. Its threshold (0.35) is fit on the char-ngram model's dev scores and applied to the encoder average. |
| S4.3: B1 threshold transport to the 49% quantile, 0.7552 | `taskB_prompts/b1_threshold_transport_49pct.py`; final build `taskB_prompts/b1_final_build.py` |
| S4.3: balanced retrain, Spearman 0.946 | `src/modal_taskb.py::sweep_balanced`, then `taskB_prompts/probes/b1_balanced_ensemble_probe.py` |
| S4.3: 18-model rank average, 0.7381 | `taskB_prompts/probes/b1_rank_average_18model.py` |
| S6: sequential-id leak | `id_leak_audit.py` |

The A2 score sequence in S5, in order:

| A2 | Split | Code |
|---|---|---|
| 0.2482 | dev_test | `history/taskA_charngram_v1/taska_baseline.py` |
| 0.2690 | dev_test | `python -m src.taska_baseline` |
| 0.3650 | dev_test | `./run_vlm_pipeline.sh v2` (uses `src/vlm_decode.py`) |
| 0.3694 | dev_test | `taskA_memes/a_devtest_rank_decode_v2.py` |
| 0.3913 | dev_test | `python -m src.vlm_ensemble --tags v2 s43 s44 s45 s46 --target dev_test --a1-count 175` |
| 0.3739 | test | `taskA_memes/a_test_decode_5seed_published_prior.py` |
| 0.4167 | test | `taskA_memes/a2_test_decode_support_corrected.py` |
| 0.4194 | test | `taskA_memes/a_test_decode_8seed_final.py` |

The 0.3694, 0.3913 and 0.3739 decodes use dev_test per-class counts read from
the platform's scoring output (0.3739 scales them to estimate test counts). The
0.4167 and 0.4194 decodes use test per-class counts read the same way, with
top-k = max(3, round(1.1 x count)).

Final runs:

| Run | Score | Built by |
|---|---|---|
| A1 881206 | 0.7889 | `taskA_memes/a_test_decode_8seed_final.py`, then `src/build_submissions.py --target test` |
| A2 881205 | 0.4194 | same as A1 |
| B1 881203 | 0.7552 | `taskB_prompts/b1_final_build.py` |
| B2 881204 | 0.7695 | `src/taskb_test_submit.py` |

`audit/` holds the original scripts behind the 169 near-duplicate pairs
(`near_duplicates_cross_split.py`, with a threshold sweep in
`near_duplicates_sensitivity.py`), the AraSafe overlap (`arasafe_overlap.py`)
and the id leak (`repro_id_leak.py`, the 30 Jul original of
`id_leak_audit.py`). They read the public data directly. The 49% B1 balance was
worked out by hand from the accuracy the platform printed (S4.3).

### Setup

Task data is not redistributed here (CC BY-NC terms). Get it from the
organizers: https://github.com/araieval/ArGuard-2026-tasks. Place it as:

```
data/taskA/splits/{train,dev,dev_test,test}.jsonl   # taskA/data/download_data.py writes these
data/taskB/train.csv
data/taskB/dev_with_label.csv
data/taskB/dev_without_label.csv
data/taskB/Arguard_binary_test_no_label.csv
data/taskB/Arguard_granular_test_no_label.csv
starter-kit/   # a clone of the organizers' repo; run_vlm_pipeline.sh uses its format checker
```

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p predictions submissions
modal token new
modal secret create huggingface HF_TOKEN=...   # or set ARGUARD_MODAL_SECRET to your secret's name
```

Training runs on Modal. The VLM container reads `QCRI/MemeLens-VLM` and the
`QCRI/ArGuard-Task1` dataset from Hugging Face. The Task B container gets
`data/taskB/` uploaded. Both use the Modal volume `arguard-cache`.

### Task A

```
# one run; the paper's runs are --seed 42 --tag v2, --seed 43..46 --tag s43..s46,
# and repeat runs --seed 42 --tag v3td, --seed 43 --tag s43td, --seed 44 --tag s44td
ARGUARD_GPU=H100 modal run --detach src/modal_vlm.py::train --seed 42 --tag v2
ARGUARD_GPU=H100 modal run src/modal_vlm.py::predict --tag v2 --split test
PYTHONPATH=. python taskA_memes/a_test_decode_8seed_final.py
python -m src.build_submissions --run-id awn_vlm_ens8 --target test
```

The 8-seed decode needs test predictions for all eight tags.
`src/build_submissions.py` reads `data/taskB/dev_without_label.csv` even with
`--target test`. With `--target devtest` it also packages B1/B2 from
`predictions/b1_dev.json` and `b2_dev.json`, so run
`taskB_prompts/taskb_persist_charngram_dev_preds.py` or `src.taskb_ensemble`
first. `run_vlm_pipeline.sh` expects the virtual environment at `.venv`.

### Task B

```
modal run src/modal_taskb.py::sweep             # writes predictions/taskb_test/
python -m src.taskb_test_submit                 # B2 file (and a threshold-0.35 B1 file)
PYTHONPATH=. python taskB_prompts/b1_final_build.py   # final B1 file
```

Dev phase: copy `history/taskB_devphase/modal_taskb.py` over
`src/modal_taskb.py`, run `modal run src/modal_taskb.py::sweep` (writes
`predictions/taskb_encoders/`), then `python -m src.taskb_ensemble` (B1 0.9542,
B2 0.9064). Files in `history/` are drop-in replacements for the `src/` module
of the same name.

### Leak audit

Run the audit against the official CSVs:

```
python id_leak_audit.py path/to/train.csv path/to/dev_with_label.csv
```

## Provenance

The code was lost in a workspace incident in August 2026. It was recovered
verbatim from the development session logs. Each module was rebuilt by
replaying the logged file writes, edits and patch scripts in order. For every
`src/` module except `scorer_b.py`, the compiled bytecode of the original
survived, and the rebuilt module compiles to the same code objects, with the
same recorded source size. Scripts that were run inline (piped into Python)
were copied unchanged from the logged command, with a new header comment.

Changes made for this release, with no logic changes:

- The Modal secret name in `src/modal_vlm.py`, `src/modal_taskb.py` and
  `history/taskB_devphase/modal_taskb.py` now comes from `ARGUARD_MODAL_SECRET`
  (default `huggingface`). The original named a secret from another project.
  The two `modal_taskb.py` files gained `import os` for this.
- The recovery header comments of the inline scripts were replaced.

| File | Status |
|---|---|
| `src/taska_baseline.py`, `a2_decode_ablation.py`, `vlm_decode.py`, `vlm_ensemble.py` | verbatim; matches the surviving bytecode |
| `src/taskb_baseline.py`, `taskb_ensemble.py`, `taskb_test_submit.py` | verbatim; matches the surviving bytecode |
| `src/submit.py`, `build_submissions.py`, `__init__.py` | verbatim; matches the surviving bytecode |
| `src/modal_vlm.py`, `src/modal_taskb.py` | verbatim except the secret name; matched the bytecode before that change |
| `src/scorer_b.py` | verbatim from one logged write; never run, no bytecode to check |
| `run_vlm_pipeline.sh` | verbatim from one logged write; no bytecode for a shell script |
| `taskA_memes/*.py`, `taskB_prompts/**/*.py` | body verbatim from the logged inline runs; header comment new |
| `history/taskA_charngram_v1/taska_baseline.py` | verbatim; the module before its one logged edit |
| `history/taskB_devphase/modal_taskb.py` | verbatim except the secret name; the module as first written |
| `id_leak_audit.py` | written for the first release on 31 Aug 2026 |
| `audit/repro_id_leak.py` | verbatim from one logged write (30 Jul) |
| `audit/near_duplicates_*.py`, `audit/arasafe_overlap.py` | body verbatim from the logged 29 Jul inline runs; header comment new |

The first release shipped `submit.py` and `taskb_test_submit.py` at the top
level. Those were rewritten from memory after the loss and never run. Their
bytecode does not match the surviving files. The recovered originals in `src/`
built the submitted files and replace them. In the originals, the final B1 rule
lives in `taskB_prompts/b1_final_build.py`, not in `taskb_test_submit.py`.

Gaps:

- The uploads to Codabench are not in the logs. The final run IDs are matched
  to files by score. B1 and B2 match earlier scores of the same files to every
  printed digit. A2 matches the 8-seed file (0.4194), which no other A2 file
  scored. A1 0.7889 was scored by both the 5-seed and the 8-seed files; the
  8-seed file was staged last.
- Prediction files, submission zips and Task A checkpoints are not included.
  Task B fine-tuned weights were never saved. Re-running gives new
  probabilities, so the submitted files cannot be rebuilt bit for bit.
- The three runs tagged `td` (v3td, s43td, s44td) repeat seeds 42, 43 and 44.
  They were launched with `ARGUARD_TRAIN_SPLITS=train+dev`, but
  `src/modal_vlm.py` reads that variable inside the remote Modal function,
  where local environment variables are not set. All three logged 657
  optimizer steps, the train-only count (train+dev would give 750). So all
  eight runs trained on train only, and the final A1/A2 ensemble is 8 runs over
  5 distinct seeds.
- The short commands that read per-class counts from the scoring output were
  not saved. The counts are constants in the scripts (`DEVTEST_SUPPORT` in
  `src/vlm_ensemble.py`, `SUPPORT` and `DEVTEST` in the dev_test decodes,
  `TRUE` in the test decodes). The 148/352 Task A test prior comes from the
  dataset card.
- v2 and s43 to s46 were trained by an earlier state of `src/modal_vlm.py`. It
  differs only in how `_predict` attaches the adapter and in lacking the
  train+dev switch. That state is not included.
- The Task B test sweep ran `src/modal_taskb.py` before `sweep_balanced` and
  the `balance` flag (default off) were added.
- The `src/taskb_test_submit.py` docstring says the B1 threshold is fit on the
  encoder average. The code fits it on the char-ngram model's dev scores.
- The Moroccan x adult_content guard in `src/taskb_test_submit.py` changed 0
  rows in the logged runs, so the submitted B2 is plain argmax.
- `id_leak_audit.py` was written for the first release; the original 30 Jul
  script is `audit/repro_id_leak.py`.

## Paper corrections

Recovering this code let us check the paper against what actually ran. The
corrected version (`paper/Younis_ArGuard2026_corrected.pdf`) fixes these
statements in the 10 Sep 2026 camera-ready:

- The A2 gains: switching to the VLM with the decode fixed added +0.096,
  decode changes +0.068 (0.043 of it from test-phase scoring feedback), more
  runs +0.025, and the move to the blind test cost 0.017. The camera-ready said
  about 0.15 came from decoding and under 0.03 from the model. The title now
  says "Most of What Is Left to Tune".
- The A2 score before test-phase feedback is 0.3739, not 0.3913 (a dev-test
  score).
- The final Task A ensemble is 8 runs over 5 seeds, all trained on train only;
  no train+dev runs and no out-of-fold threshold refit.
- The first B1 test file used the char-ngram member's threshold (0.35), not the
  ensemble's dev optimum.
- Smaller fixes: the decode-ablation caption, scorer precision (four decimals),
  loss weights, the Prop2Hate/ArMeme overlap, the dialect token, compute
  (about 9 H100-hours), and the removal of an error analysis no code backed.

Companion ImageEval 2026 system paper and code:
https://github.com/itsbakr/imageeval-2026

## Citing

```bibtex
@inproceedings{younis-2026-arguard,
  title     = {{Ahmed Younis} at {ArGuard Shared Tasks}: Calibration Is Most of What Is
               Left to Tune: Decode-Time Structure and Threshold Transport for
               {A}rabic Harmful-Content Detection},
  author    = {Younis, Ahmed},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing
               Conference: Shared Tasks},
  year      = {2026}, address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  pages     = {465--469},
  url       = {https://aclanthology.org/2026.arabicnlp-sharedtasks.61/}
}
```
