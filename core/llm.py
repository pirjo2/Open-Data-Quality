from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import json
import re

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)
AMBIG_BIN_RE = re.compile(r"\b(0\s*or\s*1|1\s*or\s*0)\b", re.IGNORECASE)


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
    Put output rules BEFORE the context so truncation does not remove them.
    """
    rendered = _safe_format(str(prompt_template).strip(), values)
    reminder = (
        "IMPORTANT: Output must follow the requested format exactly.\n"
        "Do not output Yes/No. Do not output placeholders.\n"
    )
    return (
        rendered.strip()
        + "\n\n"
        + reminder
        + "\n--- CONTEXT START ---\n"
        + (context or "").strip()
        + "\n--- CONTEXT END ---\n"
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str):
    """
    Transformers v5-safe runner: uses AutoModel + generate (no pipeline task strings).
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

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

            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt) :].strip()
            return text

    return runner


def _parse_json(text: str) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
    jm = JSON_OBJ_RE.search(text or "")
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
            conf_f = float(conf)
        except Exception:
            conf_f = None

    evid_s = str(evid).strip() if evid is not None else None
    return ans, conf_f, evid_s


def _extract_confidence(text: str, json_conf: Optional[float], base_conf: float) -> float:
    cm = CONF_LINE_RE.search(text or "")
    if cm:
        try:
            return max(0.0, min(1.0, float(cm.group(1))))
        except Exception:
            pass
    if json_conf is not None:
        try:
            return max(0.0, min(1.0, float(json_conf)))
        except Exception:
            pass
    return max(0.0, min(1.0, base_conf))


def _extract_evidence(text: str, json_evid: Optional[str]) -> str:
    em = EVID_LINE_RE.search(text or "")
    if em:
        return em.group(1).strip()
    return (json_evid or "").strip()


def _parse_binary(text: str, json_ans: Any) -> Tuple[Optional[float], float]:
    t = (text or "").strip()

    m = ANSWER_LINE_RE.search(t)
    if m:
        ans = m.group(1).strip()
        if AMBIG_BIN_RE.search(ans):
            return None, 0.0
        if YES_RE.search(ans) and not NO_RE.search(ans):
            return 1.0, 0.7
        if NO_RE.search(ans) and not YES_RE.search(ans):
            return 0.0, 0.7
        nm = NUM_RE.search(ans)
        if nm:
            try:
                return (1.0 if float(nm.group(0)) >= 1 else 0.0), 0.8
            except Exception:
                return None, 0.0

    if json_ans is not None:
        if isinstance(json_ans, str):
            if YES_RE.search(json_ans) and not NO_RE.search(json_ans):
                return 1.0, 0.6
            if NO_RE.search(json_ans) and not YES_RE.search(json_ans):
                return 0.0, 0.6
        try:
            return (1.0 if float(json_ans) >= 1 else 0.0), 0.7
        except Exception:
            pass

    if YES_RE.search(t) and not NO_RE.search(t):
        return 1.0, 0.5
    if NO_RE.search(t) and not YES_RE.search(t):
        return 0.0, 0.5

    if AMBIG_BIN_RE.search(t):
        return None, 0.0

    nm = NUM_RE.search(t)
    if nm:
        try:
            return (1.0 if float(nm.group(0)) >= 1 else 0.0), 0.35
        except Exception:
            return None, 0.0

    return None, 0.0


def _parse_count(text: str, json_ans: Any) -> Tuple[Optional[float], float]:
    t = (text or "").strip()

    m = ANSWER_LINE_RE.search(t)
    if m:
        ans = m.group(1).strip()
        nm = NUM_RE.search(ans)
        if nm:
            try:
                return float(nm.group(0)), 0.8
            except Exception:
                return None, 0.0

    if json_ans is not None:
        try:
            return float(json_ans), 0.7
        except Exception:
            pass

    nm = NUM_RE.search(t)
    if nm:
        try:
            return float(nm.group(0)), 0.25
        except Exception:
            return None, 0.0

    return None, 0.0


def _parse_date(text: str, json_ans: Any) -> Tuple[Optional[str], float]:
    t = (text or "").strip()

    m = ANSWER_LINE_RE.search(t)
    if m:
        ans = m.group(1).strip()
        if ans.upper() == "UNKNOWN":
            return None, 0.8
        dm = DATE_RE.search(ans)
        if dm:
            return dm.group(1), 0.85

    if json_ans is not None and isinstance(json_ans, str):
        if json_ans.strip().upper() == "UNKNOWN":
            return None, 0.7
        dm = DATE_RE.search(json_ans)
        if dm:
            return dm.group(1), 0.7

    dm = DATE_RE.search(t)
    if dm:
        return dm.group(1), 0.35

    if "UNKNOWN" in t.upper():
        return None, 0.3

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
    """
    cfg = (prompt_defs or {}).get(symbol)
    if cfg is None or hf_runner is None:
        return None, "", 0.0, ""

    typ = str(cfg.get("type", "binary")).strip()
    prompt_template = str(cfg.get("prompt", ""))

    prompt = format_prompt(
        prompt_template=prompt_template,
        context=context,
        values={**(extra_values or {}), "N": N},
    )

    def _run_once(p: str) -> Tuple[Optional[float | str], str, float, str]:
        raw = hf_runner(p, max_new_tokens=96)
        raw_str = str(raw or "").strip()

        j_ans, j_conf, j_evid = _parse_json(raw_str)
        evidence = _extract_evidence(raw_str, j_evid)

        if typ == "date":
            val, base_conf = _parse_date(raw_str, j_ans)
            conf = _extract_confidence(raw_str, j_conf, base_conf)
            return val, raw_str, conf, evidence

        if typ in ("count_0_to_N", "count"):
            val, base_conf = _parse_count(raw_str, j_ans)
            if val is None:
                return None, raw_str, 0.0, evidence
            val = max(0.0, min(float(N), float(val)))
            conf = _extract_confidence(raw_str, j_conf, base_conf)
            return float(val), raw_str, conf, evidence

        val, base_conf = _parse_binary(raw_str, j_ans)
        if val is None:
            return None, raw_str, 0.0, evidence
        conf = _extract_confidence(raw_str, j_conf, base_conf)
        return float(val), raw_str, conf, evidence

    val, raw_str, conf, evid = _run_once(prompt)

    # retry if totally unparsable (very common with T5 returning just "Yes")
    if val is None and conf == 0.0:
        retry_prefix = (
            "FORMAT ENFORCEMENT:\n"
            "Return ONLY the required lines. Example:\n"
            "answer: 1\n"
            "confidence: 0.6\n"
            "evidence: inferred: <reason>\n\n"
        )
        val2, raw2, conf2, evid2 = _run_once(retry_prefix + prompt)
        if val2 is not None or conf2 > conf:
            return val2, raw2, conf2, evid2

    return val, raw_str, conf, evid
