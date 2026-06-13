"""
The EXACT summary prompt used by Smart Resume Screener
(ai-model/app/openai_summarizer.py).

This is shared by:
  - generate_summaries.py  (teacher produces the ideal answer to THIS prompt)
  - the eventual screener integration (Phase 5)

Keeping one source of truth guarantees the fine-tuned model is a drop-in
replacement: it's trained on the same question the screener will ask at runtime.
"""


def build_summary_prompt(resume: str, job: str) -> str:
    return f"""
    You are an AI assistant tasked with evaluating a candidate's resume against a job description.

    1. Summarize the candidate's experience and qualifications.
    2. Identify their key strengths from the resume.
    3. Identify possible weaknesses or missing qualifications based on the job description.
    4. Evaluate their overall fit for the position.

    Resume:
    {resume}

    Job Description:
    {job}

    Provide a concise paragraph covering the above 4 points.
    """
