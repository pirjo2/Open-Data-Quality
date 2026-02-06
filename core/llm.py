from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import re
import json

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# numeric-only reply (prevents "AgeGroup1" -> 1 errors)
NUM_ONLY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", re.DOTALL)

# expected formats like:
# answer: 1
# evidence: ...
ANSWER_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*([01])\s*$")
EVIDENCE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

def _sanitize_text(s: str) -> str:
    """Remove control chars that often break tokenization / confuse models."""
    if s is None:
        return ""
    # Remove NULL bytes and other control chars except \n and \t
    s = s.replace("\x00", "")
    s = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    return s

def format_prompt(prompt_template: str, context: str, N: int) -> str:
    # Make output formatting stricter -> more reliable parsing
    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + _sanitize_text(context).strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n"
        + "\nIMPORTANT OUTPUT RULES:\n"
        + "1) First line must be exactly: answer: 0 or answer: 1 (or answer: YYYY-MM-DD / UNKNOWN for date tasks)\n"
        + "2) Second line must be: evidence: <short quote or none>\n"
        + "3) Do not add any other lines or text.\n"
    )

@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5 safe: use AutoModel + generate (no pipeline task strings).
    Returns a callable: runner(prompt, max_new_tokens) -> generated_text
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )

    # ✅ Use slow tokenizer to avoid "byte fallback not implemented" warning
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    kind = "seq2seq"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        kind = "causal"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    def runner(prompt: str, max_new_tokens: int = 64) -> str:
        prompt = _sanitize_text(prompt)
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            # causal models often echo prompt
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()

            return text

    return runner

def _parse_value_and_confidence(generated_text: str, typ: str) -> Tuple[Optional[float | str], float]:
    """
    Parse based on expected strict output:
      answer: ...
      evidence: ...
    """
    txt = _sanitize_text(generated_text).strip()

    # ---- date ----
    if typ == "date":
        # look for answer line first
        # allow "answer: UNKNOWN"
        m_ans = re.search(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$", txt)
        if m_ans:
            ans = m_ans.group(1).strip()
            if ans.upper() == "UNKNOWN":
                return None, 0.0
            m_date = DATE_RE.search(ans)
            if m_date:
                return m_date.group(1), 0.7

        # fallback: any date in text
        m = DATE_RE.search(txt)
        if not m:
            return None, 0.0
        return m.group(1), 0.5

    # ---- JSON (optional advanced format) ----
    jm = JSON_RE.search(txt)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            val = obj.get("value", None)
            conf = obj.get("confidence", 0.0)
            confidence = float(conf) if conf is not None else 0.0
            if val is None:
                return None, confidence
            return float(val), confidence
        except Exception:
            pass

    # ---- strict answer: 0/1 ----
    am = ANSWER_RE.search(txt)
    if am:
        return float(am.group(1)), 0.7

    # ---- numeric-only fallback (rare) ----
    nm = NUM_ONLY_RE.match(txt)
    if nm:
        return float(nm.group(1)), 0.5

    return None, 0.0

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[float | str], str, float]:
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg["prompt"], context, N)

    generated_text = hf_runner(prompt, max_new_tokens=64)

    parsed_val, conf = _parse_value_and_confidence(generated_text, typ)

    # If no parse, return 0 for numeric types
    if parsed_val is None:
        if typ in ("binary", "count_0_to_N", "count"):
            return 0.0, generated_text, 0.0
        return None, generated_text, 0.0

    if typ == "binary":
        v = 1.0 if float(parsed_val) >= 1 else 0.0
        return v, generated_text, conf

    if typ in ("count_0_to_N", "count"):
        v = max(0.0, min(float(N), float(parsed_val)))
        return v, generated_text, conf

    return float(parsed_val), generated_text, conf
