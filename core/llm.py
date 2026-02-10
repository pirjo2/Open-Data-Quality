from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import json
import re

# --- Regexes for parsing model output ---
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    """
    Format a template with placeholders like {dataset_description}.
    Missing keys are replaced by empty strings instead of raising.
    """
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    try:
        return template.format_map(_Safe(values))
    except Exception:
        return template


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any]) -> str:
    """
    Render the user-defined prompt template and attach the tabular context.
    """
    rendered = _safe_format(prompt_template.strip(), values)
    return (
        rendered
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n"
        + "\nReturn ONLY the answer in the requested format.\n"
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5-safe runner: we avoid task strings and use generate() directly.

    Returns a callable:
        runner(prompt: str, max_new_tokens: int = 96) -> str
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kind = "seq2seq"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception:
        kind = "causal"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    def runner(prompt: str, max_new_tokens: int = 96) -> str:
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

            # Causal models often echo the prompt; strip it if needed.
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner


# ------------ Parsers for different symbol types ------------


def _parse_binary(text: str) -> Tuple[Optional[float], float]:
    """
    Parse a binary answer 0/1 from the model output.
    Returns (value_or_None, confidence 0..1).
    """
    # Prefer explicit "answer:" line
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        nm = NUM_RE.search(ans)
        if nm:
            try:
                v = float(nm.group(0))
                return (1.0 if v >= 1 else 0.0), 0.7
            except Exception:
                return None, 0.0
        if YES_RE.search(ans):
            return 1.0, 0.6
        if NO_RE.search(ans):
            return 0.0, 0.6
        return None, 0.0

    # Fallback: look for yes/no in whole text
    if YES_RE.search(text) and not NO_RE.search(text):
        return 1.0, 0.5
    if NO_RE.search(text) and not YES_RE.search(text):
        return 0.0, 0.5

    nm = NUM_RE.search(text)
    if nm:
        try:
            v = float(nm.group(0))
            return (1.0 if v >= 1 else 0.0), 0.5
        except Exception:
            return None, 0.0

    return None, 0.0


def _parse_count(text: str) -> Tuple[Optional[float], float]:
    """
    Parse a non-negative count from the model output.
    """
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        nm = NUM_RE.search(ans)
        if nm:
            try:
                return float(nm.group(0)), 0.7
            except Exception:
                return None, 0.0

    nm = NUM_RE.search(text)
    if nm:
        try:
            return float(nm.group(0)), 0.5
        except Exception:
            return None, 0.0

    return None, 0.0


def _parse_date(text: str) -> Tuple[Optional[str], float]:
    """
    Parse a date in YYYY-MM-DD or UNKNOWN from the model output.
    """
    m = ANSWER_LINE_RE.search(text)
    if m:
        ans = m.group(1).strip()
        if ans.upper() == "UNKNOWN":
            return None, 0.7
        dm = DATE_RE.search(ans)
        if dm:
            return dm.group(1), 0.7

    dm = DATE_RE.search(text)
    if dm:
        return dm.group(1), 0.5

    if "UNKNOWN" in text.upper():
        return None, 0.5

    return None, 0.0


def _parse_json_block(text: str) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
    """
    Try to parse a JSON object from the output, e.g.
      {"answer": 1, "confidence": 0.8, "evidence": "..."}
    """
    jm = JSON_OBJ_RE.search(text)
    if not jm:
        return None, None, None
    try:
        obj = json.loads(jm.group(0))
    except Exception:
        return None, None, None

    ans = obj.get("answer", obj.get("value", None))
    conf = obj.get("confidence", None)
    evid = obj.get("evidence", None)

    try:
        conf_f = float(conf) if conf is not None else None
    except Exception:
        conf_f = None

    evid_s = str(evid).strip() if evid is not None else None
    return ans, conf_f, evid_s


def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
    extra_values: Dict[str, Any],
) -> Tuple[Optional[float | str], str, float, str]:
    """
    Run the LLM for one symbol.

    Returns:
        value_or_none:
            - float in [0,1] for binary
            - float for count
            - string "YYYY-MM-DD" for date
        raw_text: full raw model output
        confidence: 0..1 (heuristic)
        evidence: extracted evidence string (may be "")
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = str(cfg.get("type", "binary"))

    prompt = format_prompt(
        prompt_template=str(cfg.get("prompt", "")),
        context=context,
        values={**extra_values, "N": N},
    )

    raw = hf_runner(prompt, max_new_tokens=96)
    raw_str = str(raw or "").strip()

    # Evidence (best effort)
    em = EVID_LINE_RE.search(raw_str)
    evidence = em.group(1).strip() if em else ""

    # Try JSON block
    j_ans, j_conf, j_evid = _parse_json_block(raw_str)
    if j_evid and not evidence:
        evidence = j_evid

    cm = CONF_LINE_RE.search(raw_str)
    conf = None
    if cm:
        try:
            conf = float(cm.group(1))
        except Exception:
            conf = None
    if conf is None:
        conf = j_conf

    # --- Date type ---
    if typ == "date":
        val, base_conf = _parse_date(raw_str)
        if val is None and isinstance(j_ans, str):
            dm = DATE_RE.search(j_ans)
            if dm:
                val = dm.group(1)
        final_conf = float(conf) if conf is not None else base_conf
        return val, raw_str, max(0.0, min(1.0, final_conf)), evidence

    # --- Count type ---
    if typ in ("count_0_to_N", "count"):
        val, base_conf = _parse_count(raw_str)
        if val is None and j_ans is not None:
            try:
                val = float(j_ans)
            except Exception:
                val = None
        if val is None:
            return None, raw_str, 0.0, evidence
        # Clamp to [0, N] if applicable
        val = max(0.0, min(float(N), float(val)))
        final_conf = float(conf) if conf is not None else base_conf
        return float(val), raw_str, max(0.0, min(1.0, final_conf)), evidence

    # --- Binary (default) ---
    val, base_conf = _parse_binary(raw_str)
    if val is None and j_ans is not None:
        try:
            v = float(j_ans)
            val = 1.0 if v >= 1 else 0.0
        except Exception:
            if isinstance(j_ans, str) and YES_RE.search(j_ans):
                val = 1.0
            elif isinstance(j_ans, str) and NO_RE.search(j_ans):
                val = 0.0
            else:
                val = None

    if val is None:
        return None, raw_str, 0.0, evidence

    final_conf = float(conf) if conf is not None else base_conf
    return float(val), raw_str, max(0.0, min(1.0, final_conf)), evidence
