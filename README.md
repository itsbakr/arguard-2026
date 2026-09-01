# Ahmed Younis at ArGuard Shared Tasks (ArabicNLP 2026)

Code for the system-description paper *"Calibration Is Most of What You Tune:
Decode-Time Structure and Threshold Transport for Arabic Harmful-Content
Detection"* (ArabicNLP 2026 Shared Tasks track, co-located with EMNLP 2026,
Budapest). Team **Ahmed Younis**, Codabench username `ahmedbakr`.

Official results: **1st** on subtask A2 (fine-grained hateful memes, macro-F1
0.4194), **3rd** on A1 (binary memes, 0.7889), **3rd** on B2 (harm domain,
0.7695), 7th on B1 (safe/unsafe, 0.7552).

## Contents

```
id_leak_audit.py       # reproduces the Task B sequential-id label leak from the
                       # public dev-phase CSVs alone: contiguous per-label id runs,
                       # a nearest-id lookup fitted on train scoring 1.0000 dev
                       # macro-F1 with zero characters of text read, and the
                       # shuffled-id randomization control that collapses it to
                       # chance. Disclosed to the organizers 29 Jul 2026 with this
                       # reproduction; the test release was re-indexed in response.
submit.py              # builds and validates all four Codabench submission
                       # formats (the Terms tab, not the repo README, is
                       # authoritative for Task B), using zipfile so macOS
                       # metadata never breaks the archive
taskb_test_submit.py   # the Task B test decode: B1's threshold pinned to the 49%
                       # measured unsafe rate (worth +0.16 macro-F1 over the
                       # dev-fitted operating point) and B2's blended argmax
```

Run the audit against the official CSVs:

```
python id_leak_audit.py path/to/train.csv path/to/dev_with_label.csv
```

Task data is not redistributed here (CC BY-NC terms); fetch it from the official
ArGuard 2026 release. The Task A VLM training code was lost in a workspace
incident before the camera-ready; the decode ablation is fully specified in the
paper (Table 1 and Section 4), and this repository contains the original audit,
submission, and Task B decode tooling.

Companion ImageEval 2026 system paper and code:
https://github.com/itsbakr/imageeval-2026

## Citing

```bibtex
@inproceedings{younis-2026-arguard,
  title     = {{Ahmed Younis} at {ArGuard Shared Tasks}: Calibration Is Most of What You
               Tune: Decode-Time Structure and Threshold Transport for {A}rabic
               Harmful-Content Detection},
  author    = {Younis, Ahmed},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing
               Conference: Shared Tasks},
  year      = {2026}, address = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```
