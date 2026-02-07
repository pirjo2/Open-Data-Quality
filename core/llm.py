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
OUTPUT_FORMAT_RE = re.compile(r"(?is)\n\s*output\s+format\s*:\s*\n")  # to strip "Output format:" blocks


def _clip_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _strip_output_format_block(prompt_template: str) -> str:
    """
    Many prompt packs contain:
      Output format:
        answer: ...
        evidence: ...
    Flan-T5 often echoes that text back ("1 or 0", "YYYY-MM-DD").
    We strip it to reduce template-echo failures.
    """
    base = (prompt_template or "").strip()
    m = OUTPUT_FORMAT_RE.search(base)
    if m:
        return base[: m.start()].strip()
    return base


def format_prompt(prompt_template: str, context: str, N: int) -> str:
    base = _strip_output_format_block(prompt_template)
    ctx = _clip_text(context, 2500)

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
    Local Hugging Face runner (CPU/GPU).
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
                num_beams=1,
            )

            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

            # causal models sometimes echo prompt
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()

            return text

    return runner


def _parse_json_answer(generated_text: str) -> Tuple[Optional[Any], str, float]:
    s = (generated_text or "").strip()
    try:
        obj = json.loads(s)
        ans = obj.get("answer", None)
        ev = (obj.get("evidence", "") or "").strip()
        conf = obj.get("confidence", 0.0)
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        conf_f = max(0.0, min(1.0, conf_f))
        return ans, ev, conf_f
    except Exception:
        return None, "", 0.0


def _fallback_parse_lines(generated_text: str) -> Tuple[Optional[str], str, float]:
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
    Returns parsed value or None (= did not work)
    """
    if ans is None:
        return None

    # date
    if typ == "date":
        if isinstance(ans, str):
            s = ans.strip()
            if s.upper() in ("UNKNOWN", "NULL", "NONE"):
                return None
            m = DATE_RE.search(s)
            if not m:
                return None
            return m.group(1)
        return None

    # numeric / binary / count
    if isinstance(ans, bool):
        ans = 1 if ans else 0

    if isinstance(ans, (int, float)):
        val = float(ans)
    elif isinstance(ans, str):
        s = ans.strip()
        # reject typical junk
        if " or " in s.lower():
            return None
        if "yyyy" in s.lower():
            return None
        nm = STRICT_NUM_RE.match(s)
        if not nm:
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
        if val >= 0.5:
            return 1.0
        return 0.0

    if typ == "count_0_to_N":
        return float(max(0.0, min(float(N), val)))

    # allow floats (0.14 etc)
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
        ans_s, ev2, conf2 = _fallback_parse_lines(raw)
        ans = ans_s
        if not ev:
            ev = ev2
        if conf <= 0.0:
            conf = conf2

    parsed = _strict_parse_value(ans, typ, N)
    if parsed is None:
        return None, ev, 0.0, raw

    # If model didn't provide confidence, assign a sensible default
    if conf <= 0.0:
        conf = 0.6
        if ev and ev.lower() != "none":
            conf = 0.75

    return parsed, ev, float(conf), raw
