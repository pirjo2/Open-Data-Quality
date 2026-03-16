from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Callable
import json
import os
import re
import time

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b", re.I)
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true|present|exists|available)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false|missing|absent|not available)\b", re.IGNORECASE)


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

    def runner(prompt: str, max_new_tokens: int = 40) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    max_output_tokens=max_new_tokens,
                )

                text = getattr(response, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

                # Kui output_text on tühi, tagasta kogu response debugiks
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

    return runner


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

    raw = llm_runner(prompt, 128)

    import json
    try:
        data = json.loads(raw)
    except Exception:
        return {}, raw

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

def infer_manual_metadata_symbols(
    text: str,
    llm_runner,
) -> Tuple[Dict[str, Any], str]:
    """
    Interpret free-text metadata and return a Vetrò symbol dict.

    Returns:
        (parsed_symbols_dict, raw_output)
    """
    if not llm_runner or not text or not text.strip():
        return {}, ""

    prompt = f"""
You are extracting structured metadata for Vetrò-style open data quality evaluation.

Read the metadata text below and return ONLY valid JSON.

Rules:
- Use only these keys if evidence exists:
  pb, t, d, dc, cv, l, id, s, dp, sd, edp, ed, cd, lu, du, c
- For presence-type fields use:
  1 = present
  0 = missing
- For date fields (dp, sd, edp, ed, cd), return ISO format YYYY-MM-DD when clearly present.
- Do not invent values.
- If a field is not clearly supported by the text, omit it.
- Return ONLY JSON, no explanation.

Metadata text:
\"\"\"
{text}
\"\"\"

Example output:
{{
  "t": 1,
  "d": 1,
  "s": 1,
  "lu": 1,
  "du": 1,
  "id": 1,
  "cv": 1,
  "sd": "2022-10-03",
  "edp": "2023-10-01"
}}
"""

    try:
        raw = llm_runner(prompt, 256)
    except Exception as e:
        return {}, f"LLM_ERROR: {e}"

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}, raw

        allowed_keys = {
            "pb", "t", "d", "dc", "cv", "l", "id", "s",
            "dp", "sd", "edp", "ed", "cd", "lu", "du", "c"
        }

        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in allowed_keys:
                continue

            if k in {"dp", "sd", "edp", "ed", "cd"}:
                if isinstance(v, str) and DATE_RE.search(v):
                    cleaned[k] = DATE_RE.search(v).group(1)
                continue

            if v in [0, 1]:
                cleaned[k] = float(v)

        return cleaned, raw

    except Exception:
        return {}, raw