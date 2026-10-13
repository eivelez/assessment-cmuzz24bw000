"""Validator Agent — coherence checks for datasets before they reach the UI.

Business rules enforced in this iteration:
    1. Net sales can never be negative.
    2. Applied discounts cannot exceed 100%.
    3. Transaction dates cannot be in the future.

Backend switch
--------------
``VALIDATION_BACKEND`` selects the engine:

    - ``"deterministic"`` (default): pure vectorised pandas checks. These rules
      are hard invariants, so this path is exact, ~ms latency and fully
      testable. This is what should ship for this iteration.
    - ``"llm"``: delegates the judgement to Claude with a minimal payload,
      concise prompt and bounded token budget. Kept behind the flag so the
      same agent can grow into fuzzy/semantic rules later without rewiring.

Fallback policy (LLM path)
--------------------------
A timeout or transient LLM error must NOT take down the main request. When the
LLM call fails we fall back to the deterministic engine (correct and free) and
flag the result as ``degraded`` so the orchestrator can mark it ``suspect``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pandas as pd

from app.services.llm import anthropic_client
from app.utils.dispatcher import dispatcher
from app.utils.logger import logger

# --- Configuration -----------------------------------------------------------

# Flip to "llm" (or set the env var) to route validation through Claude.
VALIDATION_BACKEND = os.getenv("VALIDATION_BACKEND", "deterministic").lower()

# Hard cap on the LLM call so the added p95 overhead stays under ~2s.
LLM_TIMEOUT_SECONDS = 1.5
# Keep the payload minimal: only a bounded sample of rows is sent.
LLM_MAX_ROWS = 50
# Structured, short response — small token budget keeps latency predictable.
LLM_MAX_TOKENS = 256

# Discount is assumed to be expressed as a percentage (0-100). Flip to 1.0 if
# the pipeline ever stores it as a fraction.
DISCOUNT_MAX = 100.0

# Legacy data is inconsistent and bilingual — match columns by candidate names.
NET_SALES_COLUMNS = (
    "net_sales",
    "ventas_netas",
    "ventas_netas_reales",
    "net_sales_amount",
    "monto_neto",
)
DISCOUNT_COLUMNS = (
    "discount",
    "discount_pct",
    "discount_percent",
    "descuento",
    "descuento_pct",
    "descuento_aplicado",
)
DATE_COLUMNS = (
    "transaction_date",
    "fecha_transaccion",
    "fecha",
    "date",
    "transaction_ts",
)

# --- Helpers -----------------------------------------------------------------


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the actual column name matching a candidate (case-insensitive)."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion for messy legacy data.

    Strips currency/percent/thousand separators when values arrive as strings.
    Unparseable values become NaN (skipped by the checks) instead of crashing.
    """
    if series.dtype == object:
        series = (
            series.astype(str)
            .str.strip()
            .str.replace(r"[,$%\s]", "", regex=True)
            .replace({"": None, "nan": None, "none": None, "None": None})
        )
    return pd.to_numeric(series, errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """Parse dates in mixed formats; unparseable values become NaT."""
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def _examples(series: pd.Series, mask: pd.Series, limit: int = 3) -> list:
    return series[mask].head(limit).tolist()


# --- Deterministic engine ----------------------------------------------------


def _validate_deterministic(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []

    # Rule 1: net sales never negative.
    col = _find_column(df, NET_SALES_COLUMNS)
    if col is not None:
        values = _coerce_numeric(df[col])
        mask = values < 0
        if mask.any():
            errors.append(
                {
                    "rule": "net_sales_non_negative",
                    "column": col,
                    "violations": int(mask.sum()),
                    "examples": _examples(values, mask),
                }
            )

    # Rule 2: discount cannot exceed 100%.
    col = _find_column(df, DISCOUNT_COLUMNS)
    if col is not None:
        values = _coerce_numeric(df[col])
        mask = values > DISCOUNT_MAX
        if mask.any():
            errors.append(
                {
                    "rule": "discount_within_limit",
                    "column": col,
                    "violations": int(mask.sum()),
                    "examples": _examples(values, mask),
                }
            )

    # Rule 3: transaction date not in the future.
    col = _find_column(df, DATE_COLUMNS)
    if col is not None:
        values = _coerce_datetime(df[col])
        now = pd.Timestamp.now(tz="UTC")
        mask = values > now
        if mask.any():
            errors.append(
                {
                    "rule": "transaction_date_not_future",
                    "column": col,
                    "violations": int(mask.sum()),
                    "examples": [str(v) for v in values[mask].head(3)],
                }
            )

    return {"valid": not errors, "errors": errors}


# --- LLM engine --------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You are a strict data validator. Check the dataset (CSV) against these "
    "business rules:\n"
    "1) net sales must be >= 0\n"
    "2) discount must be <= 100 (percent)\n"
    "3) transaction date must not be in the future\n"
    'Respond with ONLY compact JSON: {"valid": bool, "errors": '
    '[{"rule": str, "detail": str}]}. No prose, no markdown.'
)


def _minimal_payload(df: pd.DataFrame) -> str:
    """Only the relevant columns, capped to LLM_MAX_ROWS, as compact CSV."""
    cols = [
        c
        for c in (
            _find_column(df, NET_SALES_COLUMNS),
            _find_column(df, DISCOUNT_COLUMNS),
            _find_column(df, DATE_COLUMNS),
        )
        if c is not None
    ]
    subset = df[cols] if cols else df
    return subset.head(LLM_MAX_ROWS).to_csv(index=False)


def _llm_call_blocking(system_prompt: str, user_message: str) -> str:
    """Synchronous Claude call, run in a worker thread so it can be timed out."""
    message = anthropic_client.client.messages.create(
        model=anthropic_client.default_model,
        max_tokens=LLM_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


async def _validate_llm(df: pd.DataFrame) -> dict[str, Any]:
    payload = _minimal_payload(df)
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(_llm_call_blocking, _LLM_SYSTEM_PROMPT, payload),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        parsed = json.loads(raw)
        return {
            "valid": bool(parsed.get("valid", True)),
            "errors": list(parsed.get("errors", [])),
        }
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
        # Fail-open: never tank the main request. Fall back to the exact
        # deterministic engine and flag the result as degraded.
        logger.warning(
            "LLM validation failed (%s); falling back to deterministic engine.",
            exc,
        )
        result = _validate_deterministic(df)
        result["degraded"] = True
        return result


# --- Public entrypoint -------------------------------------------------------


@dispatcher.register_validator(name="business_rules_validator")
async def validate_business_rules(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the dataset against business rules.

    Returns a dict with ``valid`` (bool) and ``errors`` (list); the LLM path may
    also add ``degraded`` (bool) when it fell back after a transient failure.
    """
    if df is None or df.empty:
        return {"valid": True, "errors": []}

    if VALIDATION_BACKEND == "llm":
        return await _validate_llm(df)
    return _validate_deterministic(df)
