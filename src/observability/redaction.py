"""Mandatory PII/secret redaction and allowlisted structured log sanitization."""

from __future__ import annotations

import re
from typing import Any

ALLOWLISTED_LOG_FIELDS: frozenset[str] = frozenset(
    {
        "trace_id",
        "thread_id",
        "step_name",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "tool_name",
        "status",
        "retry_count",
        "timestamp",
        "mode",
        "reason",
        "error_code",
    }
)

_SECRET_RE = re.compile(
    r"(?:"
    r"\bAIza[0-9A-Za-z_-]{20,}\b"
    r"|\bsk-[A-Za-z0-9_-]{16,}\b"
    r"|\bghp_[A-Za-z0-9]{20,}\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
    r"|\bBearer\s+[A-Za-z0-9._-]{12,}\b"
    r"|(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9._-]{8,}['\"]?"
    r")",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?)?\d{4,5}[\s.-]\d{4}\b|\+91[\s-]?\d{10}\b|\b[6-9]\d{9}\b"
)
_CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_AADHAAR_RE = re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_CEP_RE = re.compile(r"\b\d{5}-\d{3}\b")
_ADDRESS_RE = re.compile(
    r"\b(?:Rua|Avenida|Av\.|Alameda|Travessa|Street|St\.|Ave\.|Road|Rd\.)\s+[A-Za-z0-9À-ÿ\s.,-]{3,50}(?:,\s*\d+)?",
    re.IGNORECASE,
)


def redact_text(value: str) -> str:
    """Mask secrets, emails, phone numbers, IDs, cards, and street/postal addresses."""
    if not value:
        return value
    text = _SECRET_RE.sub("[REDACTED_SECRET]", value)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _CPF_RE.sub("[REDACTED_PII]", text)
    text = _AADHAAR_RE.sub("[REDACTED_PII]", text)
    text = _PAN_RE.sub("[REDACTED_PII]", text)
    text = _CEP_RE.sub("[REDACTED_PII]", text)
    text = _ADDRESS_RE.sub("[REDACTED_PII]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _CARD_RE.sub("[REDACTED_PII]", text)
    return text


def sanitize_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Filter payload to allowlisted keys only and redact all string/nested values."""
    sanitized: dict[str, Any] = {}
    for key in ALLOWLISTED_LOG_FIELDS:
        if key not in payload:
            continue
        val = payload[key]
        if isinstance(val, str):
            sanitized[key] = redact_text(val)
        elif isinstance(val, dict):
            sanitized[key] = {
                str(k): redact_text(v) if isinstance(v, str) else v for k, v in val.items()
            }
        else:
            sanitized[key] = val
    return sanitized


redact_log_record = sanitize_log_payload
