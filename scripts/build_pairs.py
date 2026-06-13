"""
Step 1 of dataset building: sample real (resume, job description, fit label) pairs.

Source: cnamuangtoun/resume-job-description-fit  (~8k real, pre-paired rows on HF)
We sample a balanced set across the three fit labels so the dataset contains
obvious fits, obvious non-fits, and borderline cases (the roadmap's requirement).

Why the HTTP API instead of the `datasets` library:
the HF datasets-server exposes rows over plain HTTP, so we avoid installing
`datasets`/`pyarrow` (which may lack wheels on very new Python versions).

Output: data/pairs.jsonl  — one JSON object per line:
    {"resume": "...", "jd": "...", "label": "Good Fit"}

Resumes/JDs are length-capped here so the teacher (Step 2), the training inputs
(Phase 2-3), and the eventual screener inference all see the SAME truncation.
Consistency across teacher/train/inference is what makes the distilled model behave.
"""

import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "cnamuangtoun/resume-job-description-fit"
SPLIT = "train"
PER_LABEL = 120                      # 120 x 3 labels = 360 pairs
LABELS = ["No Fit", "Potential Fit", "Good Fit"]

# The dataset is grouped by label (verified by probing). Rather than scan all
# ~6.2k rows (which trips the datasets-server rate limit), we fetch two small
# windows inside each label's region for diversity. 60 rows x 2 windows = 120.
# Windows: (offset, count).
WINDOWS = {
    "No Fit":        [(200, 60), (2500, 60)],     # region ~0-3300
    "Potential Fit": [(3450, 60), (4300, 60)],    # region ~3400-4800
    "Good Fit":      [(4950, 60), (5900, 60)],    # region ~4900-6240
}

# Length caps (characters). Resumes lead with summary/skills/recent roles, so
# truncating the tail keeps the most fit-relevant content. ~4 chars/token, so
# these caps keep total input well under a 2048-token training window.
RESUME_CAP = 4000   # ~1000 tokens
JD_CAP = 2500       # ~625 tokens

OUT = Path(__file__).resolve().parent.parent / "data" / "pairs.jsonl"
PAGE = 100          # max rows per datasets-server request


def fetch_page(offset, length):
    url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET}&config=default&split={SPLIT}"
        f"&offset={offset}&length={length}"
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)["rows"]
        except Exception as e:
            wait = 8 * (attempt + 1)
            print(f"  retry {attempt+1} at offset {offset} (wait {wait}s): {e}")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch offset {offset}")


def main():
    buckets = defaultdict(list)
    for label in LABELS:
        for offset, count in WINDOWS[label]:
            rows = fetch_page(offset, count)
            for r in rows:
                row = r["row"]
                # trust the window mapping, but double-check the label matches
                if row["label"] != label:
                    continue
                buckets[label].append(
                    (row["resume_text"], row["job_description_text"])
                )
            print(f"  {label}: fetched {len(rows)} at offset {offset}")
            time.sleep(4)  # stay under the rate limit

    print("\nCollected per label:", {k: len(v) for k, v in buckets.items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("w") as f:
        for label in LABELS:
            pool = buckets.get(label, [])
            n = min(PER_LABEL, len(pool))
            for resume, jd in pool[:n]:
                f.write(json.dumps({
                    "resume": resume[:RESUME_CAP].strip(),
                    "jd": jd[:JD_CAP].strip(),
                    "label": label,
                }) + "\n")
                written += 1
            print(f"  {label}: wrote {n}")

    print(f"\nWrote {written} pairs -> {OUT}")


if __name__ == "__main__":
    main()
