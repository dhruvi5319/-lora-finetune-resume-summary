"""
Step 3 of dataset building: split data/dataset.jsonl into train / eval.

Eval is held out and balanced across fit labels (10 per label = 30) so the
Phase 4 comparison (base vs GPT-3.5 vs fine-tuned) covers obvious fits,
non-fits, and borderline cases evenly. The model never sees eval during training.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dataset.jsonl"
TRAIN = ROOT / "data" / "train.jsonl"
EVAL = ROOT / "data" / "eval.jsonl"
EVAL_PER_LABEL = 10


def main():
    by_label = defaultdict(list)
    for line in SRC.open():
        ex = json.loads(line)
        by_label[ex.get("label", "?")].append(ex)

    train, evalset = [], []
    for label, items in by_label.items():
        evalset += items[:EVAL_PER_LABEL]
        train += items[EVAL_PER_LABEL:]

    with TRAIN.open("w") as f:
        for ex in train:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")
    with EVAL.open("w") as f:
        for ex in evalset:
            f.write(json.dumps(ex) + "\n")  # keep label in eval for analysis

    print(f"train: {len(train)} -> {TRAIN}")
    print(f"eval:  {len(evalset)} -> {EVAL}")
    print("eval per label:", {k: min(EVAL_PER_LABEL, len(v)) for k, v in by_label.items()})


if __name__ == "__main__":
    main()
