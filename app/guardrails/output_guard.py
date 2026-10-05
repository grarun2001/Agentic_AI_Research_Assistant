import re
from typing import List, Optional
from pydantic import BaseModel, Field

class OutputValidationResult(BaseModel):
    is_valid: bool
    sanitized_response: str
    flags: List[str] = Field(default_factory=list)
    warning: Optional[str] = None

# Secret / API key patterns to prevent data leakage in responses
SECRET_PATTERNS = [
    (r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_GEMINI_KEY]"),
    (r"tvly-[0-9A-Za-z-_]{30,}", "[REDACTED_TAVILY_KEY]"),
    (r"(?i)api[_-]?key\s*[:=]\s*['\"][0-9a-zA-Z\-_]{16,}['\"]", "[REDACTED_API_KEY]"),
]

def validate_output_response(
    response: Optional[str],
    internal_sources_count: int = 0,
    external_sources_count: int = 0,
    route: str = "rag"
) -> OutputValidationResult:
    """
    Validates, scrubs, and verifies the generated output from the agent workflow.
    Ensures safe response delivery, protects against secret leakage, and adds
    groundedness disclaimers if no sources were retrieved.
    """
    flags: List[str] = []

    # 1. Empty Response Check
    if not response or not response.strip():
        return OutputValidationResult(
            is_valid=False,
            sanitized_response="I was unable to retrieve sufficient grounded evidence to answer your query. Please review your document selection or rephrase your question.",
            flags=["EMPTY_RESPONSE"],
            warning="Model returned an empty response."
        )

    cleaned = response.strip()

    # 2. Secret Redaction (Prevent accidental API key leakage)
    for pattern, replacement in SECRET_PATTERNS:
        if re.search(pattern, cleaned):
            flags.append("SECRET_LEAKAGE_REDACTED")
            cleaned = re.sub(pattern, replacement, cleaned)

    # 3. Groundedness / Fallback Verification
    # If RAG was used but 0 internal documents were found and response claims definitive policy
    warning = None
    if route in ("rag", "both") and internal_sources_count == 0:
        if "policy" in cleaned.lower() or "handbook" in cleaned.lower():
            disclaimer = "\n\n> **Note:** *No matching internal company documents were retrieved for this specific question. The above information may be general guidance rather than certified internal policy.*"
            if disclaimer.strip() not in cleaned:
                cleaned += disclaimer
                flags.append("GROUNDEDNESS_DISCLAIMER_APPENDED")
                warning = "No internal documents retrieved; disclaimer appended to avoid policy hallucination."

    return OutputValidationResult(
        is_valid=True,
        sanitized_response=cleaned,
        flags=flags,
        warning=warning
    )
