from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Callable
import json
import os
import re
import time


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*[:=]\s*(.+?)\s*$")
CONF_LINE_RE = re.compile(r"(?im)^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")
EVID_LINE_RE = re.compile(r"(?im)^\s*evidence\s*[:=]\s*(.*)\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
YES_RE = re.compile(r"\b(yes|true|present|exists|available)\b", re.IGNORECASE)
NO_RE = re.compile(r"\b(no|false|missing|absent|not available)\b", re.IGNORECASE)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    try:
        return template.format_map(_SafeDict(values or {}))
    except Exception:
        return template


def format_prompt(prompt_template: str, context: str, values: Dict[str, Any]) -> str:
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
    max_retries: int = 2,
) -> Callable[[str, int], str]:
    from openai import OpenAI

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OpenAI API key is missing.")

    client = OpenAI(api_key=resolved_api_key)

    def runner(prompt: str, max_new_tokens: int = 128) -> str:
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    max_output_tokens=max_new_tokens,
                )
                text = (response.output_text or "").strip()
                if text:
                    return text
                return ""
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break

        raise RuntimeError(f"OpenAI request failed: {last_error}")

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
        return None, "", None, ""

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

    # Conservative fallback confidence
    if confidence is None and value is not None:
        confidence = 0.5

    return value, raw, confidence, evidence