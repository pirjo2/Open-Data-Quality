from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple, Optional
import re

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extract_first_number(text: str):
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if not m:
        return None
    s = m.group(0)
    return float(s) if "." in s else int(s)


def format_prompt(prompt_template: str, context: str, N: int) -> str:
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "Return ONLY the answer in the requested format. No extra text."
    )


@lru_cache(maxsize=2)
def get_hf_pipe(model_name: str):
    # Import lazily so running without LLM deps still works.
    from transformers import pipeline  # type: ignore
    # Streamlit Cloud is typically CPU
    return pipeline("text2text-generation", model=model_name, device=-1)


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_pipe,
) -> Tuple[Optional[float | str], Optional[str]]:
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_pipe is None:
        return None, None

    prompt = format_prompt(cfg["prompt"], context, N)

    out = hf_pipe(
        prompt,
        truncation=True,
        max_new_tokens=12,
        do_sample=False,
        temperature=0.0,
    )

    # Extract generated text
    generated_text = ""
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        generated_text = str(out[0].get("generated_text", "")).strip()
    else:
        generated_text = str(out).strip()

    typ = cfg.get("type", "binary")

    if typ == "date":
        m = DATE_RE.search(generated_text)
        if not m or generated_text.upper().strip() == "UNKNOWN":
            return None, generated_text
        return m.group(1), generated_text

    num = extract_first_number(generated_text)
    if num is None:
        return None, generated_text

    val = float(num)

    if typ == "binary":
        val = 1.0 if val >= 1 else 0.0
        return val, generated_text

    if typ == "count_0_to_N":
        val = max(0.0, min(float(N), val))
        return val, generated_text

    return val, generated_text