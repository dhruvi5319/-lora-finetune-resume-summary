---
base_model: meta-llama/Llama-3.2-3B-Instruct
library_name: peft
license: llama3.2
tags:
  - lora
  - qlora
  - peft
  - resume-screening
  - summarization
  - distillation
pipeline_tag: text-generation
---

# Llama 3.2 3B — Resume Fit-Summary (LoRA)

A QLoRA adapter for `meta-llama/Llama-3.2-3B-Instruct` that generates a structured 4-point
**resume↔job-description fit summary**: (1) experience summary, (2) key strengths,
(3) weaknesses/missing qualifications, (4) overall fit verdict.

Built to replace the GPT-3.5 API call in
[Smart Resume Screener](https://github.com/dhruvi5319/smart-resume-screener) with a free, local,
consistent model. Trained by **distillation** from a GPT-4-class teacher on the screener's exact prompt.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base, device_map="auto"),
    "dhruvi5319/llama-3.2-3b-resume-fit-summary",
).eval()

prompt = """You are an AI assistant tasked with evaluating a candidate's resume against a job description.
1. Summarize the candidate's experience and qualifications.
2. Identify their key strengths from the resume.
3. Identify possible weaknesses or missing qualifications based on the job description.
4. Evaluate their overall fit for the position.

Resume:
{resume}

Job Description:
{job}

Provide a concise paragraph covering the above 4 points."""

msgs = [{"role": "user", "content": prompt.format(resume=RESUME, job=JD)}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                             return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=300, do_sample=False)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

- **Method:** QLoRA (4-bit NF4 base + LoRA r=16, α=32) with **completion-only loss**.
- **Data:** 471 distilled (resume+JD → ideal summary) examples — real pairs from
  `cnamuangtoun/resume-job-description-fit` plus synthetic strong-fit augmentation; teacher = `gpt-4o-mini`.
- **Schedule:** 3 epochs, LR 2e-4 cosine, effective batch 16, on a single A100.

## Evaluation (held-out, n=30, GPT-4-class judge, position-randomized)

| | Base Llama 3.2 3B | This adapter |
|---|---|---|
| Head-to-head wins | 1 | **6** (23 ties) |
| Covers all 4 points | 28/30 | 29/30 |
| Grounded (no invented facts) | 30/30 | 30/30 |
| Avg length (teacher ≈ 900 chars) | 1214 | **904** |

## Limitations & intended use

- Scoped to English resume/JD fit summaries in the 4-point format above.
- The base model is already a capable summarizer, so gains are in **consistency, conciseness, and
  format-adherence**, not raw capability.
- May reflect biases in the underlying resume dataset; **not** a substitute for human review in hiring.
