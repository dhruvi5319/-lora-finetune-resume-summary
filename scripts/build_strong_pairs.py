"""
Augmentation: generate synthetic STRONG-FIT pairs to fix the coverage gap.

The real dataset (cnamuangtoun/...) contains almost no genuine strong matches,
so the distilled model would never learn to describe an excellent candidate.
Here we take real JDs we already have and, for each, generate a realistic resume
deliberately tailored to be a strong fit. These pairs are appended to
data/pairs.jsonl with label "Synthetic Strong" (kept for provenance), then the
normal generate_summaries.py run produces their summaries (which should now come
back as fit=strong from the teacher's independent evaluation).

Usage:
    python scripts/build_strong_pairs.py            # ~50 pairs
    python scripts/build_strong_pairs.py --n 60
    python scripts/generate_summaries.py            # then summarize the new pairs
    python scripts/split_dataset.py                 # re-split
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_summaries import load_env, make_teacher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "pairs.jsonl"
RESUME_CAP = 4000

RESUME_PROMPT = """Write a realistic resume for a candidate who is an EXCELLENT, clearly strong fit for the job below.

Requirements for the resume:
- Match or exceed the role's required years of experience and seniority.
- Explicitly demonstrate the specific skills, tools, and domain the JD names.
- 2-3 work experiences with concrete, quantified accomplishments relevant to the role.
- Realistic style: plausible company names, job titles, dates, and an education section.
- Do NOT include placeholder fields like "[Your Name]" or "[Address]". Use a realistic
  name and omit contact details entirely.
- Do not mention the job description or that this is tailored.

Output only the resume text.

Job description:
{jd}
"""


def existing_jds():
    """Return (unused_unique_jds, count_already_used_for_strong)."""
    used = set()        # JDs already turned into synthetic-strong pairs
    all_seen, jds = set(), []
    for line in PAIRS.open():
        p = json.loads(line)
        jd = p["jd"]
        key = jd[:120]
        if p.get("label") == "Synthetic Strong":
            used.add(key)
        if key not in all_seen:
            all_seen.add(key)
            jds.append((key, jd))
    unused = [jd for key, jd in jds if key not in used]
    return unused, len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="how many strong pairs")
    args = ap.parse_args()

    load_env()
    summarize, who = make_teacher()
    print(f"Teacher: {who}")

    jds, already = existing_jds()
    print(f"{already} JDs already used for strong pairs; {len(jds)} unused available")
    # spread across the unused JD pool for role diversity
    step = max(1, len(jds) // args.n)
    chosen = jds[::step][: args.n]
    print(f"Generating {len(chosen)} strong resumes")

    written = 0
    with PAIRS.open("a") as f:
        for i, jd in enumerate(chosen, 1):
            try:
                resume = summarize(RESUME_PROMPT.format(jd=jd), max_tokens=800)
            except Exception as e:
                print(f"  [{i}] FAILED: {e}")
                time.sleep(5)
                continue
            f.write(json.dumps({
                "resume": resume[:RESUME_CAP].strip(),
                "jd": jd,
                "label": "Synthetic Strong",
            }) + "\n")
            f.flush()
            written += 1
            if i % 10 == 0:
                print(f"  [{i}/{len(chosen)}] generated")
            time.sleep(0.5)

    print(f"\nAppended {written} strong pairs -> {PAIRS}")


if __name__ == "__main__":
    main()
