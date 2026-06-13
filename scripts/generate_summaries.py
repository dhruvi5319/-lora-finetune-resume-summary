"""
Step 2 of dataset building: distillation.

For each (resume, jd) pair in data/pairs.jsonl, ask a strong "teacher" model
(Claude by default) to produce the ideal fit summary, using the SAME prompt the
screener sends. The result is a chat-format training example:

    {"messages": [
        {"role": "user",      "content": <screener summary prompt>},
        {"role": "assistant", "content": <ideal summary from teacher>}
     ],
     "label": "poor"}

The student (Llama 3.2 3B) will later be fine-tuned to reproduce the assistant
turn given the user turn — i.e. to mimic the teacher.

Fit labels: the SOURCE dataset's labels are noisy, so we don't trust them. We
ask the teacher to also emit its own fit verdict (strong/partial/poor) used only
to build a balanced eval set. That verdict is teacher-only scaffolding: the
stored USER turn is the exact screener prompt (no fit instruction), so the model
stays a drop-in replacement; we strip the verdict out of the stored summary.

Usage:
    # test on a few first (recommended):
    python scripts/generate_summaries.py --limit 5
    # full run:
    python scripts/generate_summaries.py

Requires a teacher key in .env: ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY.
The script is resumable — rerunning skips pairs already summarized.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_template import build_summary_prompt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data" / "pairs.jsonl"
OUT = ROOT / "data" / "dataset.jsonl"

# Teacher model strings — change if your account exposes different ones.
ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"   # GPT-4-class, cheap
MAX_TOKENS = 450               # paragraph + the FIT line

# Teacher-only scaffolding appended to the screener prompt so we can derive a
# trustworthy fit label. NOT stored in the training example's user turn.
FIT_INSTRUCTION = (
    "\n\nAfter the paragraph, on a final separate line, output exactly one of: "
    "'FIT: strong', 'FIT: partial', or 'FIT: poor' reflecting the overall fit."
)


def parse_fit(text):
    """Pull the FIT verdict out and return (clean_summary, fit_level)."""
    fit = "partial"
    m = re.search(r"FIT:\s*(strong|partial|poor)", text, re.IGNORECASE)
    if m:
        fit = m.group(1).lower()
    # remove the FIT line from the stored summary
    clean = re.sub(r"\n?\s*FIT:\s*(strong|partial|poor)\s*$", "", text,
                   flags=re.IGNORECASE).strip()
    return clean, fit


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v:                      # last non-empty value wins; ignore blanks
                    os.environ[k] = v


def make_teacher():
    """Return a callable summary(prompt)->str using whichever key is present."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        client = Anthropic()
        model = ANTHROPIC_MODEL

        def summarize(prompt, max_tokens=MAX_TOKENS):
            r = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text.strip()

        return summarize, f"anthropic:{model}"

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        model = OPENAI_MODEL

        def summarize(prompt, max_tokens=MAX_TOKENS):
            r = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()

        return summarize, f"openai:{model}"

    sys.exit("No teacher key found. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env")


def already_done():
    """Set of resume-prefixes already summarized, so reruns resume."""
    done = set()
    if OUT.exists():
        for line in OUT.open():
            try:
                ex = json.loads(line)
                # key on the FULL prompt — the prompt PREFIX is identical for
                # every example (boilerplate), so a prefix key would over-skip.
                done.add(ex["messages"][0]["content"])
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process first N (0=all)")
    args = ap.parse_args()

    load_env()
    summarize, who = make_teacher()
    print(f"Teacher: {who}")

    pairs = [json.loads(l) for l in PAIRS.open()]
    if args.limit:
        pairs = pairs[: args.limit]

    done = already_done()
    written = skipped = failed = 0
    with OUT.open("a") as f:
        for i, p in enumerate(pairs, 1):
            prompt = build_summary_prompt(p["resume"], p["jd"])   # exact screener prompt
            if prompt in done:
                skipped += 1
                continue
            try:
                raw = summarize(prompt + FIT_INSTRUCTION)          # teacher scaffolding
            except Exception as e:
                print(f"  [{i}/{len(pairs)}] FAILED: {e}")
                failed += 1
                time.sleep(5)
                continue
            summary, fit = parse_fit(raw)
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": prompt},           # no fit instruction stored
                    {"role": "assistant", "content": summary},
                ],
                "label": fit,                                      # our derived verdict
                "source_label": p["label"],                        # keep noisy original for comparison
            }) + "\n")
            f.flush()
            written += 1
            if i % 10 == 0 or args.limit:
                print(f"  [{i}/{len(pairs)}] fit={fit} (src={p['label']}): {summary[:70]}...")
            time.sleep(0.5)

    print(f"\nDone. wrote={written} skipped={skipped} failed={failed} -> {OUT}")


if __name__ == "__main__":
    main()
