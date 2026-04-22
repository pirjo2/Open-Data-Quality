from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Callable
import json
import os
import re
import time
import yaml

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b", re.I)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true|present|exists|available)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false|missing|absent|not available)\b", re.IGNORECASE)
DEBUG_PRINT_PROMPTS = False


def parse_llm_json_loose(raw_text: str) -> Dict[str, Any]:
    """
    Parse either:
    1) one valid JSON object
    2) multiple JSON objects concatenated together

    Merge __evidence__ and __confidence__ maps when needed.
    """
    if not raw_text or not str(raw_text).strip():
        return {}

    raw_text = str(raw_text).strip()

    # First: normal JSON
    try:
        data = json.loads(raw_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Second: multiple JSON objects in one output
    decoder = json.JSONDecoder()
    idx = 0
    objects = []

    while idx < len(raw_text):
        while idx < len(raw_text) and raw_text[idx].isspace():
            idx += 1
        if idx >= len(raw_text):
            break

        try:
            obj, end_idx = decoder.raw_decode(raw_text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1

    if not objects:
        return {}

    merged: Dict[str, Any] = {}
    for obj in objects:
        for key, value in obj.items():
            if key in {"__evidence__", "__confidence__"} and isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged[key] = value

    return merged


def extract_symbols_from_realistic_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    low = text.lower()

    # ---------- test 2: updates ----------
    if re.search(r"updates\s*:\s*\[\s*\]", low):
        out["lu"] = 0.0
        out["du"] = 0.0

    if "update history" in low or "updated regularly" in low:
        out["lu"] = 1.0

    if re.search(r"updated on\s+\d{4}-\d{2}-\d{2}", low):
        out["du"] = 1.0

    # ---------- test 4: delay in publication ----------
    m = re.search(
        r"covers the period from\s+(20\d{2}-\d{2}-\d{2})\s+to\s+(20\d{2}-\d{2}-\d{2})",
        low
    )
    if m:
        out["sd"] = m.group(1)
        out["edp"] = m.group(2)

    m = re.search(r"published on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["dp"] = m.group(1)

    # ---------- test 5: delay after expiration ----------
    m = re.search(r"expired on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["ed"] = m.group(1)

    m = re.search(r"became available on\s+(20\d{2}-\d{2}-\d{2})", low)
    if m:
        out["cd"] = m.group(1)

    # ---------- test 9: eGMS positive / negative statements ----------
    if re.search(r"^title\s*:", text, re.I | re.M):
        out["t"] = 1.0
    elif "title is missing" in low:
        out["t"] = 0.0

    if re.search(r"^description\s*:", text, re.I | re.M):
        out["d"] = 1.0
    elif "description is missing" in low:
        out["d"] = 0.0

    if re.search(r"^identifier\s*:", text, re.I | re.M):
        out["id"] = 1.0
    elif "no identifier" in low or "identifier is missing" in low:
        out["id"] = 0.0

    if re.search(r"^publisher\s*:", text, re.I | re.M):
        out["pb"] = 1.0
    elif "publisher is missing" in low:
        out["pb"] = 0.0

    if re.search(r"^coverage\s*:", text, re.I | re.M):
        out["cv"] = 1.0
    elif "no coverage information" in low:
        out["cv"] = 0.0

    if re.search(r"^language\s*:", text, re.I | re.M):
        out["l"] = 1.0
    elif "language is missing" in low:
        out["l"] = 0.0

    if re.search(r"^source\s*:", text, re.I | re.M):
        out["s"] = 1.0
    elif "no source information" in low:
        out["s"] = 0.0

    if re.search(r"^date of creation\s*:\s*(20\d{2}-\d{2}-\d{2})", text, re.I | re.M):
        out["dc"] = 1.0
    elif "no creation date" in low or "creation date is missing" in low:
        out["dc"] = 0.0

    if re.search(r"^category\s*:", text, re.I | re.M):
        out["c"] = 1.0
    elif "no category" in low or "category is missing" in low:
        out["c"] = 0.0

    return out

def _safe_format(template: str, values: Dict[str, Any]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    try:
        return template.format_map(_SafeDict(values or {}))
    except Exception:
        return template


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any]) -> str:
    if not isinstance(prompt_template, str):
        prompt_template = str(prompt_template)

    rendered = _safe_format(prompt_template.strip(), values)
    return (
        rendered
        + "\n\n--- CONTEXT START ---\n"
        + context.strip()
        + "\n--- CONTEXT END ---\n\n"
        + "Return exactly 3 lines and nothing else:\n"
        + "answer: <value>\n"
        + "confidence: <0..1>\n"
        + "evidence: <short reason>\n"
    )


@lru_cache(maxsize=4)
def get_hf_runner(model_name: str) -> Callable[[str, int], str]:
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

    def runner(prompt: str, max_new_tokens: int = 128) -> str:
        with torch.no_grad():
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            if kind == "causal" and text.startswith(prompt):
                text = text[len(prompt):].strip()
            return text

    return runner

def get_openai_runner(
    model_name: str,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> Callable[[str, int], str]:
    from openai import OpenAI

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key is missing.")

    client = OpenAI(api_key=resolved_api_key)

    def _extract_text_from_response(response) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "message":
                    continue
                for part in getattr(item, "content", []) or []:
                    if getattr(part, "type", None) in {"output_text", "text"}:
                        txt = getattr(part, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()
        except Exception:
            pass

        return ""

    def runner(prompt: str, max_new_tokens: int = 128) -> str:
        is_gpt5_family = str(model_name).lower().startswith("gpt-5")

        kwargs = {
            "model": model_name,
            "input": prompt.strip(),
            "max_output_tokens": max_new_tokens,
        }

        if is_gpt5_family:
            kwargs["reasoning"] = {"effort": "minimal"}
            kwargs["text"] = {"verbosity": "low"}

        response = client.responses.create(**kwargs)
        text = _extract_text_from_response(response)
        if text:
            return text

        try:
            return f"EMPTY_OUTPUT | FULL_RESPONSE: {response.model_dump_json(indent=2)}"
        except Exception:
            return f"EMPTY_OUTPUT | RESPONSE_REPR: {response!r}"

    return runner

'''def get_openai_runner(
    model_name: str,
    api_key: Optional[str] = None,
    max_retries: int = 1,
) -> Callable[[str, int], str]:
    from openai import OpenAI

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key is missing.")

    client = OpenAI(api_key=resolved_api_key)

    def _extract_text_from_response(response) -> str:
        # 1) easiest path
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # 2) inspect output items
        try:
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) != "message":
                    continue
                for part in getattr(item, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type in {"output_text", "text"}:
                        txt = getattr(part, "text", None)
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()
        except Exception:
            pass

        return ""

    def runner(prompt: str, max_new_tokens: int = 128) -> str:
        last_error: Optional[Exception] = None
        compact_prompt = prompt.strip()

        # GPT-5-family often burns output on reasoning unless we constrain it.
        is_gpt5_family = str(model_name).lower().startswith("gpt-5")

        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "input": compact_prompt,
                    "max_output_tokens": max_new_tokens,
                }

                if is_gpt5_family:
                    kwargs["reasoning"] = {"effort": "minimal"}
                    kwargs["text"] = {"verbosity": "low"}

                response = client.responses.create(**kwargs)

                text = _extract_text_from_response(response)
                if text:
                    return text

                # if incomplete due to token budget, retry once with stricter prompt
                try:
                    status = getattr(response, "status", None)
                    incomplete_details = getattr(response, "incomplete_details", None)
                    reason = None
                    if incomplete_details is not None:
                        reason = getattr(incomplete_details, "reason", None)
                        if reason is None and isinstance(incomplete_details, dict):
                            reason = incomplete_details.get("reason")

                    if status == "incomplete" and reason == "max_output_tokens" and attempt < max_retries:
                        compact_prompt = (
                            "Return ONLY a compact JSON object. "
                            "No explanation, no markdown, no extra text.\n\n"
                            + prompt.strip()
                        )
                        continue
                except Exception:
                    pass

                try:
                    return f"EMPTY_OUTPUT | FULL_RESPONSE: {response.model_dump_json(indent=2)}"
                except Exception:
                    return f"EMPTY_OUTPUT | RESPONSE_REPR: {response!r}"

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                return f"OPENAI_ERROR: {e}"

        return f"OPENAI_ERROR: {last_error}"

    return runner'''


def get_llm_runner(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> Callable[[str, int], str]:
    provider = (provider or "").strip().lower()

    if provider == "openai":
        return get_openai_runner(model_name=model_name, api_key=api_key)

    if provider == "huggingface":
        return get_hf_runner(model_name=model_name)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _parse_answer_value(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None

    lower = text.lower()

    if DATE_RE.search(text):
        return DATE_RE.search(text).group(1)

    if YES_RE.search(lower):
        return 1.0
    if NO_RE.search(lower):
        return 0.0

    try:
        return float(text)
    except Exception:
        pass

    m = NUM_RE.search(text)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass

    return text


def _extract_structured_response(raw: str) -> Tuple[Any, Optional[float], str]:
    raw = (raw or "").strip()
    if not raw:
        return None, None, ""

    answer = None
    confidence = None
    evidence = ""

    # 1) Preferred format: answer/confidence/evidence lines
    ans_match = ANSWER_LINE_RE.search(raw)
    conf_match = CONF_LINE_RE.search(raw)
    evid_match = EVID_LINE_RE.search(raw)

    if ans_match:
        answer = _parse_answer_value(ans_match.group(1).strip())

    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except Exception:
            confidence = None

    if evid_match:
        evidence = evid_match.group(1).strip()

    if answer is not None:
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
        return answer, confidence, evidence

    # 2) JSON fallback
    json_match = JSON_OBJ_RE.search(raw)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            answer = _parse_answer_value(str(obj.get("answer", "")).strip())
            conf_raw = obj.get("confidence")
            evidence = str(obj.get("evidence", "")).strip()

            if conf_raw is not None:
                confidence = float(conf_raw)
                confidence = max(0.0, min(1.0, confidence))

            return answer, confidence, evidence
        except Exception:
            pass

    # 3) Fallback: try whole output as answer
    answer = _parse_answer_value(raw)
    return answer, confidence, evidence

def infer_symbols_batch(symbols, context, prompt_defs, llm_runner):
    symbol_list = "\n".join(symbols)

    prompt = f"""
You are evaluating metadata of an open dataset.

Context:
{context}

Return values for these symbols:

{symbol_list}

Respond ONLY in JSON.
"""
    if DEBUG_PRINT_PROMPTS:
        print("\n--- MANUAL_METADATA PROMPT START ---\n")
        print(prompt)
        print("\n--- MANUAL_METADATA PROMPT END ---\n")
    raw = llm_runner(prompt, 128)
    data = parse_llm_json_loose(raw)
    return data, raw

def infer_symbol(
    symbol: str,
    context: str,
    N: int,
    prompt_defs: Optional[Dict[str, Any]],
    llm_runner: Optional[Callable[[str, int], str]],
    extra_values: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, Optional[float], str]:
    """
    Returns:
        value, raw_output, confidence, evidence
    """
    if not llm_runner or not prompt_defs:
        return None, "", None, ""

    prompt_template = prompt_defs.get(symbol)
    if not prompt_template:
        return None, f"NO_PROMPT_FOUND_FOR_{symbol}", None, ""

    if isinstance(prompt_template, dict):
        prompt_template = (
            prompt_template.get("prompt")
            or prompt_template.get("template")
            or prompt_template.get("text")
        )

    if not isinstance(prompt_template, str) or not prompt_template.strip():
        return None, f"INVALID_PROMPT_FOR_{symbol}: {prompt_defs.get(symbol)}", None, ""

    values = {"N": N}
    if extra_values:
        values.update(extra_values)

    prompt = format_prompt(
        prompt_template=prompt_template,
        context=context,
        values=values,
    )

    try:
        raw = llm_runner(prompt, 128)
    except Exception as e:
        return None, f"LLM_ERROR: {e}", None, ""

    value, confidence, evidence = _extract_structured_response(raw)

    if confidence is None and value is not None:
        confidence = 0.5

    return value, raw, confidence, evidence

def get_prompt_template_with_fallback(
    prompts_cfg: Dict[str, Any],
    regime: str,
    prompt_name: str,
    fallback_template: str,
) -> Tuple[str, str]:
    """
    Return (template, source), where source is:
    - 'yaml:<regime>' if template was found from prompts.yaml
    - 'fallback' otherwise
    """
    try:
        tmpl = (
            (prompts_cfg or {})
            .get("prompt_regimes", {})
            .get(regime, {})
            .get(prompt_name, None)
        )
        if isinstance(tmpl, str) and tmpl.strip():
            return tmpl, f"yaml:{regime}"
    except Exception:
        pass

    return fallback_template, "fallback"


def _split_symbol_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Any]]:
    if not isinstance(data, dict):
        return {}, {}, {}

    evidence_map = data.get("__evidence__", {})
    confidence_map = data.get("__confidence__", {})

    if not isinstance(evidence_map, dict):
        evidence_map = {}
    if not isinstance(confidence_map, dict):
        confidence_map = {}

    symbol_data = {
        k: v for k, v in data.items()
        if not str(k).startswith("__")
    }
    return symbol_data, evidence_map, confidence_map

def infer_manual_metadata_symbols(
    text: str,
    llm_runner,
    prompts_cfg: Optional[Dict[str, Any]] = None,
    prompt_regime: str = "zero_shot",
) -> Tuple[Dict[str, Any], str, str, Dict[str, Any]]:
    if not llm_runner or not text or not text.strip():
        return {}, "", "not_used", {"evidence": {}, "confidence": {}, "parsed": {}}

    fallback_prompt = """
Extract Vetrò metadata symbols from the text below.

Return ONLY valid JSON.
Allowed keys:
pb, t, d, dc, cv, l, id, s, c, dp, sd, edp, ed, cd, lu, du

Output structure:
{
  "<symbol>": <value>,
  "__evidence__": {"<symbol>": "<short evidence string>"},
  "__confidence__": {"<symbol>": <0..1>}
}

Rules:
- Presence keys must be 0 or 1
- Date keys must be in YYYY-MM-DD format
- Natural-language update frequency counts as lu=1
- du=1 only if update dates are explicitly given
- If uncertain, omit the key
- Return JSON only

Text:
{text}
"""

    prompt_template, prompt_source = get_prompt_template_with_fallback(
        prompts_cfg=prompts_cfg or {},
        regime=prompt_regime,
        prompt_name="manual_metadata_extraction",
        fallback_template=fallback_prompt,
    )

    prompt = prompt_template.format(text=text)

    try:
        raw = llm_runner(prompt, 96)
    except Exception as e:
        return {}, f"LLM_ERROR: {e}", prompt_source, {"evidence": {}, "confidence": {}, "parsed": {}}

    data = parse_llm_json_loose(raw)
    if not isinstance(data, dict) or not data:
        return {}, raw, prompt_source, {"evidence": {}, "confidence": {}, "parsed": {}}
        symbol_data, evidence_map, confidence_map = _split_symbol_payload(data)

    allowed_keys = {
        "pb", "t", "d", "dc", "cv", "l", "id", "s", "c",
        "dp", "sd", "edp", "ed", "cd", "lu", "du"
    }

    cleaned: Dict[str, Any] = {}
    cleaned_evidence: Dict[str, str] = {}
    cleaned_confidence: Dict[str, Any] = {}

    for k, v in symbol_data.items():
        if k not in allowed_keys:
            continue

        if k in {"dp", "sd", "edp", "ed", "cd"}:
            if isinstance(v, str):
                m = re.search(r"\d{4}-\d{2}-\d{2}", v)
                if m:
                    cleaned[k] = m.group(0)
        else:
            if v in [0, 1]:
                cleaned[k] = float(v)
            elif isinstance(v, (int, float)):
                cleaned[k] = float(v)

        if k in cleaned:
            evid = evidence_map.get(k)
            conf = confidence_map.get(k)
            if evid is not None:
                cleaned_evidence[k] = str(evid)
            if conf is not None:
                cleaned_confidence[k] = conf

    debug = {
        "parsed": data,
        "evidence": cleaned_evidence,
        "confidence": cleaned_confidence,
    }
    return cleaned, raw, prompt_source, debug
