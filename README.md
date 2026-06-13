# LoRA Fine-Tune: Resume Fit-Summary Generator

Fine-tuning **Llama 3.2 3B Instruct** with **QLoRA** to generate structured resume↔job-description
fit summaries — replacing the GPT-3.5 API call in
[Smart Resume Screener](https://github.com/dhruvi5319/smart-resume-screener) with a free, local,
consistent model.

🤗 **Model:** [dhruvi5319/llama-3.2-3b-resume-fit-summary](https://huggingface.co/dhruvi5319/llama-3.2-3b-resume-fit-summary)

---

## TL;DR

The screener sends each (resume, job description) pair to GPT-3.5 to produce a qualitative
4-point fit summary (experience · strengths · weaknesses · overall fit). This project distills
that behaviour into a 3B model you can run yourself — **no API key, no per-call cost, consistent
output format** — and evaluates it with a controlled A/B.

| Metric (held-out, n=30, GPT-4-class judge) | Base Llama 3.2 3B | **Fine-tuned (v2)** |
|---|---|---|
| Head-to-head wins vs base | — | **6–1** (23 ties) |
| Covers all 4 required points | 28/30 | **29/30** |
| Grounded (no invented facts) | 30/30 | 30/30 |
| Avg summary length (teacher ≈ 900 chars) | 1214 (wordy) | **904** (concise, matches teacher) |

**Honest read:** the base instruct model is already a competent summarizer, so the win is in
**consistency, conciseness, format-adherence, and API independence** — not a brand-new capability.
See [Results](#results) and [What fine-tuning did (and didn't) do](#what-fine-tuning-did-and-didnt-do).

## Approach

**Distillation.** A GPT-4-class teacher (`gpt-4o-mini`) writes the ideal 4-point summary for each
pair using the screener's *exact* prompt; the student (Llama 3.2 3B) is fine-tuned to reproduce it.

**QLoRA.** The base model is loaded in 4-bit; only a small LoRA adapter (rank 16) is trained — so
it runs on a single Colab GPU and the artifact is a ~30 MB adapter, not a 6 GB model.

## Dataset

- **Source:** [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) — real, pre-paired resumes + JDs.
- **501 examples** → 471 train / 30 held-out eval (balanced 10/10/10 across fit levels).
- **Noisy labels caught:** the source dataset's fit labels were unreliable (e.g. an Administrative
  Assistant resume tagged "Good Fit" for a Principal Data Engineer role). We **ignore them** and have
  the teacher emit its own verdict (`strong`/`partial`/`poor`) — used only to balance the eval set.
- **Strong-fit augmentation:** real data had almost no genuine strong matches, so ~140 synthetic
  strong-fit resumes were generated against real JDs (50% independently confirmed `strong` by the teacher).

## Training

QLoRA via `transformers` + `peft` + `trl`, on a Colab A100.

| Setting | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Quantization | 4-bit NF4 (double quant) |
| LoRA | r=16, α=32, dropout=0.05, all attention + MLP projections |
| LR / schedule | 2e-4, cosine, 3% warmup |
| Epochs / eff. batch | 3 / 16 |
| Loss | **completion-only** (mask the prompt; train only on the summary) — see A/B below |

### The completion-only A/B (why v2)

v1 trained on the full sequence; v2 trained **only on the assistant summary** (prompt tokens masked).
Changing that single variable improved the result:

| | v1 (full-seq loss) | **v2 (completion-only)** |
|---|---|---|
| Net win margin vs base | +3 (6–3) | **+5 (6–1)** |
| Lost to base on "strong" cases | 2 | **0** |
| Grounded / covers-4 | 30/30 · 29/30 | 30/30 · 29/30 |

## Results

- Preferred **6–1** over base Llama in blind, position-randomized head-to-head judging.
- **0 regressions** on strong-fit cases (base won 2 of those before completion-only loss).
- **Concise & consistent:** 904-char average vs base's wordy 1214, matching the teacher's style with tight variance.
- **No degenerate outputs**, **100% grounded** (no invented facts) across all 30.

### What fine-tuning did (and didn't) do

It did **not** teach Llama to summarize — it already could. It taught it to do *this specific task*
in *one consistent format and concise style, by default, locally, without the API*. Fine-tuning
shapes **behaviour/style**, not knowledge. The gain here is modest *because the base model was
already strong at the task* — fine-tuning yields larger jumps when the base is weak at the target
(strict structured output, niche domains, unusual formats).

## Using the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base),
                                  "dhruvi5319/llama-3.2-3b-resume-fit-summary")
model.eval()
# build the screener prompt, apply_chat_template(..., add_generation_prompt=True), then generate
```

## Repo structure

```
notebooks/train_lora.ipynb   # QLoRA training (Colab) — model load → LoRA → train → eval
scripts/
  build_pairs.py             # sample balanced real resume+JD pairs
  build_strong_pairs.py      # synthesize strong-fit pairs (coverage gap fix)
  generate_summaries.py      # teacher distillation -> chat dataset
  split_dataset.py           # train / eval split
  prompt_template.py         # the screener's exact summary prompt (shared)
eval/
  judge.py                   # LLM-as-judge: base vs fine-tuned, win-rate + grounding
  eval_results*.json         # base vs fine-tuned generations (v1, v2)
  judge_results*.json        # judge verdicts
data/                        # train.jsonl, eval.jsonl, ...
```

## Reproduce

```bash
pip install -r requirements.txt
cp .env.example .env                    # add HF_TOKEN + a teacher key (OPENAI_API_KEY)

python scripts/build_pairs.py           # real pairs
python scripts/build_strong_pairs.py    # + synthetic strong fits
python scripts/generate_summaries.py    # teacher distillation
python scripts/split_dataset.py         # train/eval split
# then run notebooks/train_lora.ipynb on a Colab GPU
python eval/judge.py eval/eval_results_v2.json   # evaluate
```

## Status

- [x] Phase 0 — Setup
- [x] Phase 1 — Dataset
- [x] Phase 2 — Training setup
- [x] Phase 3 — Train & debug
- [x] Phase 4 — Evaluate
- [ ] Phase 5 — Integrate into screener
- [x] Phase 6 — Publish
