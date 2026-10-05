import re
from typing import List, Optional
from pydantic import BaseModel, Field

class InputValidationResult(BaseModel):
    is_valid: bool
    sanitized_query: str
    error_message: Optional[str] = None
    flags: List[str] = Field(default_factory=list)
    risk_score: float = 0.0

# Prompt injection and jailbreak trigger patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+rules",
    r"system\s*prompt",
    r"reveal\s+(your|the)\s+(system|initial)\s+(prompt|instructions)",
    r"you\s+are\s+now\s+(in\s+)?(dan\s+mode|jailbroken|unrestricted)",
    r"bypass\s+(all\s+)?(safety|guardrails|filters)",
    r"act\s+as\s+an\s+unfiltered\s+ai",
    r"pretend\s+you\s+have\s+no\s+rules",
    r"override\s+system\s+directives"
]

# Unsafe code / script injection tags
UNSAFE_PATTERNS = [
    r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
    r"javascript\s*:",
    r"onerror\s*=",
    r"onload\s*=",
    r"eval\s*\("
]

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 2500

def validate_input_query(query: Optional[str]) -> InputValidationResult:
    """
    Validates and sanitizes a user query before feeding into agents or retrieval.
    Rejects empty, oversized, prompt injection, and malicious requests.
    """
    flags: List[str] = []

    # 1. Null / Empty Check
    if not query or not query.strip():
        return InputValidationResult(
            is_valid=False,
            sanitized_query="",
            error_message="Query cannot be empty or solely whitespace.",
            flags=["EMPTY_QUERY"],
            risk_score=1.0
        )

    # Clean query of control characters and strip whitespace
    cleaned = query.replace("\x00", "").strip()

    # 2. Length Checks
    if len(cleaned) < MIN_QUERY_LENGTH:
        return InputValidationResult(
            is_valid=False,
            sanitized_query=cleaned,
            error_message=f"Query is too short (min {MIN_QUERY_LENGTH} characters). Please ask a specific question.",
            flags=["QUERY_TOO_SHORT"],
            risk_score=0.3
        )

    if len(cleaned) > MAX_QUERY_LENGTH:
        return InputValidationResult(
            is_valid=False,
            sanitized_query=cleaned[:MAX_QUERY_LENGTH],
            error_message=f"Query exceeds the maximum allowed length of {MAX_QUERY_LENGTH} characters.",
            flags=["QUERY_TOO_LONG"],
            risk_score=0.5
        )

    # 3. Prompt Injection / Jailbreak Detection
    lower_query = cleaned.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower_query, re.IGNORECASE):
            flags.append("PROMPT_INJECTION_DETECTED")
            return InputValidationResult(
                is_valid=False,
                sanitized_query=cleaned,
                error_message="Request rejected: Input contains adversarial or unauthorized system override instructions.",
                flags=flags,
                risk_score=0.95
            )

    # 4. Malicious Script Tags Detection & Neutralization
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lower_query, re.IGNORECASE):
            flags.append("XSS_PATTERN_NEUTRALIZED")
            cleaned = re.sub(pattern, "[removed_unsafe_token]", cleaned, flags=re.IGNORECASE)

    return InputValidationResult(
        is_valid=True,
        sanitized_query=cleaned,
        error_message=None,
        flags=flags,
        risk_score=0.0 if not flags else 0.2
    )
