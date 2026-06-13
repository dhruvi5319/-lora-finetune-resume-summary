"""
Phase 4: LLM-as-judge evaluation of base vs fine-tuned fit summaries.

For each held-out example, a GPT-4-class judge sees the resume+JD and BOTH
summaries (in randomized order to avoid position bias) and decides:
  - does each cover all 4 required points?
  - is each grounded (no invented facts)?
  - which is better overall?

Outputs eval/judge_results.json + an aggregate table.
"""

import json, os, random, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# optional CLI arg: path to an eval_results*.json (defaults to v1)
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "eval_results.json"
EVAL = ROOT / "data" / "eval.jsonl"
OUT = RESULTS.with_name(RESULTS.stem.replace("eval_results", "judge_results") + ".json")
MODEL = "gpt-4o-mini"


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ[k.strip()] = v.strip()


JUDGE = """You evaluate two AI-generated "fit summaries" assessing a candidate's resume against a job description.
A good summary must: (1) summarize the candidate's experience, (2) identify key strengths,
(3) identify weaknesses/missing qualifications vs the JD, (4) give an overall fit verdict.
It should be concise, specific, and grounded in the actual resume (no invented facts).

RESUME + JOB DESCRIPTION:
{ctx}

RESPONSE A:
{a}

RESPONSE B:
{b}

Return ONLY JSON:
{{"a_covers_all_4": true/false, "b_covers_all_4": true/false,
  "a_grounded": true/false, "b_grounded": true/false,
  "winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""


def main():
    load_env()
    from openai import OpenAI
    client = OpenAI()

    results = json.load(open(RESULTS))
    evalset = [json.loads(l) for l in open(EVAL)]
    assert len(results) == len(evalset), "results/eval length mismatch"

    random.seed(42)
    out = []
    for i, (res, ex) in enumerate(zip(results, evalset)):
        ctx = ex["messages"][0]["content"][:5000]
        swap = random.random() < 0.5            # randomize A/B position
        a_is, b_is = ("finetuned", "base") if swap else ("base", "finetuned")
        a = res[a_is]
        b = res[b_is]
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": JUDGE.format(ctx=ctx, a=a, b=b)}],
            )
            j = json.loads(r.choices[0].message.content)
        except Exception as e:
            print("  fail", i, e); time.sleep(3); continue

        win = {"A": a_is, "B": b_is}.get(j.get("winner"), "tie")
        pick = lambda key, who: j[key] if who == "a" else j[key.replace("a_", "b_")]
        rec = {
            "fit_label": res["fit_label"],
            "winner": win,
            "base_covers4": j["a_covers_all_4"] if a_is == "base" else j["b_covers_all_4"],
            "ft_covers4": j["a_covers_all_4"] if a_is == "finetuned" else j["b_covers_all_4"],
            "base_grounded": j["a_grounded"] if a_is == "base" else j["b_grounded"],
            "ft_grounded": j["a_grounded"] if a_is == "finetuned" else j["b_grounded"],
            "reason": j.get("reason", ""),
        }
        out.append(rec)
        print(f"[{i+1}/{len(results)}] {res['fit_label']}: winner={win}")
        time.sleep(0.3)

    json.dump(out, open(OUT, "w"), indent=2)

    n = len(out)
    w = Counter(x["winner"] for x in out)
    print(f"\n===== JUDGE RESULTS (n={n}) =====")
    print(f"Overall winner:  fine-tuned={w['finetuned']}  base={w['base']}  tie={w['tie']}")
    print(f"Covers all 4 pts: base={sum(x['base_covers4'] for x in out)}/{n}   "
          f"fine-tuned={sum(x['ft_covers4'] for x in out)}/{n}")
    print(f"Grounded:         base={sum(x['base_grounded'] for x in out)}/{n}   "
          f"fine-tuned={sum(x['ft_grounded'] for x in out)}/{n}")
    # win rate by fit category
    for lab in ["strong", "partial", "poor"]:
        sub = [x for x in out if x["fit_label"] == lab]
        ww = Counter(x["winner"] for x in sub)
        print(f"  [{lab}] fine-tuned={ww['finetuned']} base={ww['base']} tie={ww['tie']}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
