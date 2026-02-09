from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import json
import re

# --- Patterns ---
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)

# If the model starts generating code-ish junk, treat as failure
CODEY_RE = re.compile(r"\b(for\s+\w+\s+in\s+range|input\s*\(|def\s+|return\s+|=\s*int\(|while\s+)\b", re.IGNORECASE)

# Context markers (sometimes models echo these)
CTX_START = "--- CONTEXT START ---"
CTX_END = "--- CONTEXT END ---"


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(_Safe(values))
    except Exception:
        return template


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any]) -> str:
    """
    Keep prompt template authoritative. We only append context and a small reminder.
    """
    rendered = _safe_format(prompt_template.strip(), values)
    return (
        rendered
        + "\n\n"
        + CTX_START
        + "\n"
        + context.strip()
        + "\n"
        + CTX_END
        + "\n\nReturn ONLY the answer in the requested format.\n"
    )


def _strip_echoed_context(raw: str) -> str:
    """
    If the model echoes the context, remove it before parsing.
    """
    if not raw:
        return ""

    text = raw

    # Remove any echoed context block
    start = text.find(CTX_START)
    end = text.find(CTX_END)
    if start != -1 and end != -1 and end > start:
        # Remove from CTX_START to CTX_END (inclusive)
        text = text[:start] + text[end + len(CTX_END):]

    # Also remove leftover marker lines
    text = text.replace(CTX_START, "").replace(CTX_END, "")

    return text.strip()


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5-safe runner: AutoModel + generate (no pipeline task strings).
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    def runner(prompt: str, max_new_tokens: int = 96) -> str:
        with torch.inference_mode():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            # causal models sometimes echo prompt
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()

            return text

    return runner


def _is_ambiguous_answer(ans: str) -> bool:
    """
    Catch common useless outputs like:
      - "1 or 0"
      - "0/1"
      - "yes or no"
      - "Answer: 1 or 0"
    """
    a = (ans or "").strip().lower()
    if not a:
        return True
    if "1 or 0" in a or "0 or 1" in a or "0/1" in a or "1/0" in a:
        return True
    if "yes or no" in a:
        return True
    # Contains BOTH 0 and 1 as separate tokens → likely ambiguous template
    has0 = re.search(r"\b0\b", a) is not None
    has1 = re.search(r"\b1\b", a) is not None
    if has0 and has1 and "answer" in a and "confidence" not in a:
        return True
    return False


def _parse_confidence(clean_text: str) -> Optional[float]:
    cm = CONF_LINE_RE.search(clean_text)
    if not cm:
        return None
    try:
        return max(0.0, min(1.0, float(cm.group(1))))
    except Exception:
        return None


def _parse_evidence(clean_text: str) -> str:
    em = EVID_LINE_RE.search(clean_text)
    return (em.group(1).strip() if em else "")


def _parse_json(clean_text: str) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
    jm = JSON_OBJ_RE.search(clean_text)
    if not jm:
        return None, None, None
    try:
        obj = json.loads(jm.group(0))
    except Exception:
        return None, None, None

    ans = obj.get("answer", obj.get("value", None))
    conf = obj.get("confidence", None)
    evid = obj.get("evidence", None)

    conf_f = None
    if conf is not None:
        try:
            conf_f = max(0.0, min(1.0, float(conf)))
        except Exception:
            conf_f = None

    evid_s = str(evid).strip() if evid is not None else None
    return ans, conf_f, evid_s


def _parse_binary(clean_text: str) -> Tuple[Optional[float], float]:
    """
    Returns (value or None, base_conf).
    Binary is treated as "soft": allow 0..1. Values > 1 clamp to 1.
    """
    if CODEY_RE.search(clean_text):
        return None, 0.0

    m = ANSWER_LINE_RE.search(clean_text)
    if m:
        ans = m.group(1).strip()
        if _is_ambiguous_answer(ans):
            return None, 0.0

        # yes/no
        if YES_RE.search(ans) and not NO_RE.search(ans):
            return 1.0, 0.7
        if NO_RE.search(ans) and not YES_RE.search(ans):
            return 0.0, 0.7

        nm = NUM_RE.search(ans)
        if nm:
            x = float(nm.group(0))
            # soft binary clamp
            if x < 0:
                x = 0.0
            if x > 1:
                x = 1.0
            return x, 0.75

        return None, 0.0

    # fallback: entire text
    if YES_RE.search(clean_text) and not NO_RE.search(clean_text):
        return 1.0, 0.55
    if NO_RE.search(clean_text) and not YES_RE.search(clean_text):
        return 0.0, 0.55

    nm = NUM_RE.search(clean_text)
    if nm:
        x = float(nm.group(0))
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        return x, 0.5

    return None, 0.0


def _parse_count(clean_text: str) -> Tuple[Optional[float], float]:
    if CODEY_RE.search(clean_text):
        return None, 0.0

    m = ANSWER_LINE_RE.search(clean_text)
    if m:
        ans = m.group(1).strip()
        if _is_ambiguous_answer(ans):
            return None, 0.0
        nm = NUM_RE.search(ans)
        if nm:
            return float(nm.group(0)), 0.75
        return None, 0.0

    nm = NUM_RE.search(clean_text)
    if nm:
        return float(nm.group(0)), 0.5

    return None, 0.0


def _parse_date(clean_text: str) -> Tuple[Optional[str], float]:
    if CODEY_RE.search(clean_text):
        return None, 0.0

    m = ANSWER_LINE_RE.search(clean_text)
    if m:
        ans = m.group(1).strip()
        if _is_ambiguous_answer(ans):
            return None, 0.0
        if ans.upper() == "UNKNOWN":
            return None, 0.75
        dm = DATE_RE.search(ans)
        if dm:
            return dm.group(1), 0.75
        return None, 0.0

    # fallback: but ONLY from cleaned text (context stripped)
    dm = DATE_RE.search(clean_text)
    if dm:
        return dm.group(1), 0.5

    if "UNKNOWN" in clean_text.upper():
        return None, 0.5

    return None, 0.0


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
    extra_values: Dict[str, Any],
) -> Tuple[Optional[float | str], str, float, str]:
    """
    Returns: (value_or_none, raw_text, confidence_0_to_1, evidence_string)

    IMPORTANT semantics:
      - None  => did not work
      - 0.0   => real zero (valid result)
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = cfg.get("type", "binary")

    prompt = format_prompt(
        prompt_template=str(cfg.get("prompt", "")),
        context=context,
        values={**extra_values, "N": N},
    )

    raw = hf_runner(prompt, max_new_tokens=96)
    raw_str = str(raw or "").strip()

    clean = _strip_echoed_context(raw_str)

    # evidence: either line-based, or JSON evidence fallback
    evidence = _parse_evidence(clean)

    j_ans, j_conf, j_evid = _parse_json(clean)
    if j_evid and not evidence:
        evidence = j_evid

    # confidence: explicit line wins, else JSON, else base_conf
    conf_line = _parse_confidence(clean)
    conf: Optional[float] = conf_line if conf_line is not None else j_conf

    if typ == "date":
        val, base_conf = _parse_date(clean)
        if val is None and isinstance(j_ans, str):
            dm = DATE_RE.search(j_ans)
            val = dm.group(1) if dm else None
        final_conf = conf if conf is not None else base_conf
        final_conf = max(0.0, min(1.0, float(final_conf)))
        return val, raw_str, final_conf, evidence

    if typ in ("count_0_to_N", "count"):
        val, base_conf = _parse_count(clean)
        if val is None and j_ans is not None:
            try:
                val = float(j_ans)
            except Exception:
                val = None
        if val is None:
            return None, raw_str, 0.0, evidence
        val = max(0.0, min(float(N), float(val)))
        final_conf = conf if conf is not None else base_conf
        final_conf = max(0.0, min(1.0, float(final_conf)))
        return float(val), raw_str, final_conf, evidence

    # binary (soft 0..1)
    val, base_conf = _parse_binary(clean)

    if val is None and j_ans is not None:
        try:
            x = float(j_ans)
            if x < 0:
                x = 0.0
            if x > 1:
                x = 1.0
            val = x
        except Exception:
            if isinstance(j_ans, str) and YES_RE.search(j_ans) and not NO_RE.search(j_ans):
                val = 1.0
            elif isinstance(j_ans, str) and NO_RE.search(j_ans) and not YES_RE.search(j_ans):
                val = 0.0
            else:
                val = None

    if val is None:
        return None, raw_str, 0.0, evidence

    final_conf = conf if conf is not None else base_conf
    final_conf = max(0.0, min(1.0, float(final_conf)))
    return float(val), raw_str, final_conf, evidence
