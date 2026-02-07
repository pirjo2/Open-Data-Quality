from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import json
import re

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
NUM_ANY_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
NUM_ONLY_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$", re.DOTALL)

ANSWER_NUM_RE = re.compile(r"\banswer\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)
VALUE_NUM_RE = re.compile(r"\bvalue\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)
EVIDENCE_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

YES_RE = re.compile(r"\b(yes|true|yep|present)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false|nope|absent)\b", re.IGNORECASE)
UNKNOWN_RE = re.compile(r"\b(unknown|n/?a|not\s+available|cannot\s+determine)\b", re.IGNORECASE)

AMBIGUOUS_RE = re.compile(r"\b(1\s*or\s*0|yes\s*or\s*no)\b", re.IGNORECASE)


def _extract_first_number_anywhere(text: str) -> Optional[float]:
    m = NUM_ANY_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def format_prompt(prompt_template: str, context: str, N: int, typ: str) -> str:
    """
    Adds a short few-shot constraint that tends to work better with FLAN-T5.
    """
    example = ""
    if typ == "binary":
        example = "Example output:\nanswer: 1\nevidence: none\n"
    elif typ in ("count_0_to_N", "count"):
        example = f"Example output:\nanswer: {min(3, max(0, N))}\nevidence: none\n"
    elif typ == "date":
        example = "Example output:\nanswer: 2025-10-25\nevidence: none\n"

    return (
        prompt_template.strip()
        + "\n\n--- CONTEXT START ---\n"
        + (context or "").strip()
        + "\n--- CONTEXT END ---\n"
        + f"\nN={int(N)}\n"
        + "\nReturn exactly two lines:\nanswer: <value>\nevidence: <short quote or 'none'>\n"
        + (example and ("\n" + example))
        + "Do not add any other text."
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5 safe runner: use AutoModel + generate (no pipeline task strings).
    Returns a callable: runner(prompt, max_new_tokens) -> generated_text
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kind = "seq2seq"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        kind = "causal"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    def runner(prompt: str, max_new_tokens: int = 64) -> str:
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

            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner


def _parse_value_confidence_evidence(generated_text: str, typ: str, N: int) -> Tuple[Optional[float | str], float, str]:
    """
    Returns (value, confidence, evidence).
    Confidence is heuristic unless the model returns JSON with confidence.
    """
    txt = (generated_text or "").strip()
    evidence = ""
    em = EVIDENCE_LINE_RE.search(txt)
    if em:
        evidence = em.group(1).strip()

    if AMBIGUOUS_RE.search(txt):
        return None, 0.0, evidence
    if UNKNOWN_RE.search(txt):
        return None, 0.0, evidence

    # JSON: {"value": ..., "confidence": ..., "evidence": ...}
    jm = JSON_RE.search(txt)
    if jm:
        try:
            obj = json.loads(jm.group(0))
            val = obj.get("value", None)
            conf = float(obj.get("confidence", 0.0) or 0.0)
            ev = obj.get("evidence", "")
            if isinstance(ev, str) and ev:
                evidence = ev.strip()
            if val is None:
                return None, max(0.0, min(1.0, conf)), evidence
            if typ == "date":
                if isinstance(val, str) and DATE_RE.search(val):
                    return DATE_RE.search(val).group(1), max(0.0, min(1.0, conf)), evidence
                return None, max(0.0, min(1.0, conf)), evidence
            return float(val), max(0.0, min(1.0, conf)), evidence
        except Exception:
            pass

    # Date
    if typ == "date":
        m = DATE_RE.search(txt)
        if not m:
            if "YYYY-MM-DD" in txt.upper():
                return None, 0.0, evidence
            return None, 0.0, evidence
        return m.group(1), (0.55 if evidence else 0.45), evidence

    # Binary: yes/no OR numbers anywhere
    if typ == "binary":
        has_yes = bool(YES_RE.search(txt))
        has_no = bool(NO_RE.search(txt))
        if has_yes and not has_no:
            return 1.0, (0.55 if evidence else 0.45), evidence
        if has_no and not has_yes:
            return 0.0, (0.55 if evidence else 0.45), evidence

        m = ANSWER_NUM_RE.search(txt) or VALUE_NUM_RE.search(txt)
        if m:
            v = _extract_first_number_anywhere(m.group(1))
        else:
            v = _extract_first_number_anywhere(txt)

        if v is None:
            return None, 0.0, evidence
        return (1.0 if v >= 1 else 0.0), (0.50 if evidence else 0.40), evidence

    # Count / numeric
    m = ANSWER_NUM_RE.search(txt) or VALUE_NUM_RE.search(txt)
    if m:
        v = _extract_first_number_anywhere(m.group(1))
    else:
        mo = NUM_ONLY_RE.match(txt)
        v = float(mo.group(1)) if mo else _extract_first_number_anywhere(txt)

    if v is None:
        return None, 0.0, evidence

    if typ in ("count_0_to_N", "count"):
        v = max(0.0, min(float(N), float(v)))
        return float(v), (0.50 if evidence else 0.35), evidence

    return float(v), (0.55 if evidence else 0.40), evidence


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[float | str], str, float, str]:
    """
    Returns:
      (parsed_value, evidence, confidence, raw_generated_text)

    parsed_value is None when parsing fails / answer is ambiguous.
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = str(cfg.get("type", "binary"))
    prompt = format_prompt(cfg["prompt"], context, int(N), typ)

    raw = hf_runner(prompt, max_new_tokens=64)

    val, conf, evidence = _parse_value_confidence_evidence(raw, typ=typ, N=int(N))
    return val, evidence, float(conf), str(raw or "").strip()
