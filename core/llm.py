from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import json
import re

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
STRICT_BIN_RE = re.compile(r"^\s*([01])\s*$")
STRICT_NUM_RE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*([01]|[-+]?\d+(?:\.\d+)?)\s*$")
EVIDENCE_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([-+]?\d+(?:\.\d+)?)\s*$")

def _clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."

def format_prompt(prompt_template: str, context: str, N: int) -> str:
    """
    We enforce one strict output format (JSON) at the very end.
    This reduces "1 or 0" and placeholder answers like "YYYY-MM-DD".
    """
    base = (prompt_template or "").strip()
    ctx = _clip_text(context, 3500)

    return (
        base
        + "\n\n--- CONTEXT START ---\n"
        + ctx
        + "\n--- CONTEXT END ---\n"
        + f"\nN={N}\n\n"
        + "Return EXACTLY one JSON object on a single line, no extra text:\n"
          '{"answer": <value>, "evidence": "<short quote or none>", "confidence": <0..1>}\n'
        + "Rules:\n"
        + "- For binary: answer must be 0 or 1 (number).\n"
        + "- For count: answer must be a number (0..N).\n"
        + "- For date: answer must be YYYY-MM-DD or null.\n"
        + "- If you cannot determine the answer from context, use null and confidence 0.\n"
    )

@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5 safe: use AutoModel + generate (no pipeline task strings).
    Returns callable(prompt, max_new_tokens) -> generated_text
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

    # causal models often need pad token
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def runner(prompt: str, max_new_tokens: int = 96) -> str:
        with torch.no_grad():
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

def _parse_json_answer(generated_text: str) -> Tuple[Optional[Any], str, float]:
    """
    Parse strict one-line JSON:
    {"answer": ..., "evidence": "...", "confidence": 0..1}
    """
    s = (generated_text or "").strip()
    try:
        obj = json.loads(s)
        ans = obj.get("answer", None)
        ev = obj.get("evidence", "") or ""
        conf = obj.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = max(0.0, min(1.0, conf_f))
        return ans, ev.strip(), conf_f
    except Exception:
        return None, "", 0.0

def _fallback_parse_lines(generated_text: str) -> Tuple[Optional[str], str, float]:
    """
    If model fails JSON, try:
      answer: X
      evidence: ...
      confidence: ...
    """
    s = (generated_text or "").strip()
    ans = None
    ev = ""
    conf = 0.0

    am = ANSWER_LINE_RE.search(s)
    if am:
        ans = am.group(1).strip()

    em = EVIDENCE_LINE_RE.search(s)
    if em:
        ev = (em.group(1) or "").strip()

    cm = CONF_LINE_RE.search(s)
    if cm:
        try:
            conf = float(cm.group(1))
        except Exception:
            conf = 0.0

    conf = max(0.0, min(1.0, conf))
    return ans, ev, conf

def _strict_parse_value(ans: Any, typ: str, N: int) -> Optional[Any]:
    """
    typ: binary | count_0_to_N | date | float
    Return None if invalid (== "did not work").
    """
    if ans is None:
        return None

    # date
    if typ == "date":
        if isinstance(ans, str):
            if ans.strip().upper() == "UNKNOWN":
                return None
            m = DATE_RE.search(ans.strip())
            if not m:
                return None
            return m.group(1)
        return None

    # numeric/binary/count
    if isinstance(ans, bool):
        ans = 1 if ans else 0

    if isinstance(ans, (int, float)):
        val = float(ans)
    elif isinstance(ans, str):
        s = ans.strip()
        # reject typical junk like "1 or 0" / "YYYY-MM-DD"
        if " or " in s.lower():
            return None
        if "yyyy" in s.lower():
            return None
        # strict number only
        nm = STRICT_NUM_RE.match(s)
        if not nm:
            # also allow strict binary only
            bm = STRICT_BIN_RE.match(s)
            if bm:
                val = float(bm.group(1))
            else:
                return None
        else:
            val = float(nm.group(1))
    else:
        return None

    if typ == "binary":
        if val not in (0.0, 1.0):
            # allow near 0/1 if model returns 0.0/1.0 anyway
            if val >= 0.5:
                val = 1.0
            else:
                val = 0.0
        return float(val)

    if typ == "count_0_to_N":
        val = max(0.0, min(float(N), float(val)))
        return float(val)

    return float(val)

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Dict[str, Any],
    hf_runner,
) -> Tuple[Optional[Any], str, float, str]:
    """
    Returns: (value_or_None, evidence, confidence, raw_text)
    None => did not work
    """
    cfg = prompt_defs.get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = cfg.get("type", "binary")
    prompt = format_prompt(cfg.get("prompt", ""), context, N)
    raw = hf_runner(prompt, max_new_tokens=96)
    raw = (raw or "").strip()

    ans, ev, conf = _parse_json_answer(raw)
    if ans is None:
        # fallback to line parsing
        ans_s, ev2, conf2 = _fallback_parse_lines(raw)
        ans = ans_s
        if not ev:
            ev = ev2
        if conf == 0.0:
            conf = conf2

    parsed = _strict_parse_value(ans, typ, N)

    # If parsed is None => did not work
    if parsed is None:
        return None, ev, 0.0, raw

    # If confidence missing, give a reasonable default
    if conf <= 0.0:
        conf = 0.6
        if ev and ev.lower() != "none":
            conf = 0.75

    return parsed, ev, float(conf), raw
