import pandas as pd
import pytest

import app.agents.validator as validator
from app.agents.validator import validate_business_rules


@pytest.mark.asyncio
async def test_validator_returns_expected_shape():
    """Smoke test — validator returns a dict with valid/errors keys."""
    df = pd.DataFrame()
    result = await validate_business_rules(df)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "errors" in result


@pytest.mark.asyncio
async def test_empty_dataframe_is_valid():
    result = await validate_business_rules(pd.DataFrame())
    assert result == {"valid": True, "errors": []}


@pytest.mark.asyncio
async def test_clean_dataset_is_valid():
    df = pd.DataFrame(
        {
            "net_sales": [100, 250.5, 0],
            "discount": [0, 50, 100],
            "transaction_date": ["2020-01-01", "2021-06-15", "2019-12-31"],
        }
    )
    result = await validate_business_rules(df)
    assert result["valid"] is True
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_negative_net_sales_flagged():
    df = pd.DataFrame({"net_sales": [100, -5, 20]})
    result = await validate_business_rules(df)
    assert result["valid"] is False
    rules = {e["rule"] for e in result["errors"]}
    assert "net_sales_non_negative" in rules


@pytest.mark.asyncio
async def test_discount_over_100_flagged():
    df = pd.DataFrame({"discount": [10, 150, 100]})
    result = await validate_business_rules(df)
    assert result["valid"] is False
    rules = {e["rule"] for e in result["errors"]}
    assert "discount_within_limit" in rules


@pytest.mark.asyncio
async def test_future_date_flagged():
    future = (pd.Timestamp.now() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    df = pd.DataFrame({"transaction_date": ["2020-01-01", future]})
    result = await validate_business_rules(df)
    assert result["valid"] is False
    rules = {e["rule"] for e in result["errors"]}
    assert "transaction_date_not_future" in rules


@pytest.mark.asyncio
async def test_numeric_fields_as_strings_are_coerced():
    """Legacy data: numbers arrive as strings, with currency/percent symbols."""
    df = pd.DataFrame({"net_sales": ["$1,000", "-5", "20"], "discount": ["10%", "150%", "0%"]})
    result = await validate_business_rules(df)
    rules = {e["rule"] for e in result["errors"]}
    assert "net_sales_non_negative" in rules  # "-5" detected
    assert "discount_within_limit" in rules  # "150%" detected


@pytest.mark.asyncio
async def test_mixed_date_formats_do_not_crash():
    df = pd.DataFrame(
        {"transaction_date": ["2020-01-01", "01/15/2021", "2019-12-31 10:30:00", "garbage"]}
    )
    result = await validate_business_rules(df)
    # No future dates among the parseable ones; unparseable is ignored, no crash.
    assert result["valid"] is True


@pytest.mark.asyncio
async def test_bilingual_column_names_detected():
    df = pd.DataFrame(
        {
            "ventas_netas": [-1],
            "descuento": [200],
            "fecha_transaccion": ["2020-01-01"],
        }
    )
    result = await validate_business_rules(df)
    rules = {e["rule"] for e in result["errors"]}
    assert {"net_sales_non_negative", "discount_within_limit"} <= rules


@pytest.mark.asyncio
async def test_unrelated_columns_are_ignored():
    df = pd.DataFrame({"customer_name": ["a", "b"], "region": ["north", "south"]})
    result = await validate_business_rules(df)
    assert result["valid"] is True


@pytest.mark.asyncio
async def test_llm_backend_falls_back_to_deterministic_on_error(monkeypatch):
    """A transient LLM failure must not crash — it falls back and flags degraded."""
    monkeypatch.setattr(validator, "VALIDATION_BACKEND", "llm")

    def boom(*_args, **_kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(validator, "_llm_call_blocking", boom)

    df = pd.DataFrame({"net_sales": [-10]})
    result = await validate_business_rules(df)

    assert result["degraded"] is True
    assert result["valid"] is False  # deterministic engine still caught it
    rules = {e["rule"] for e in result["errors"]}
    assert "net_sales_non_negative" in rules


@pytest.mark.asyncio
async def test_llm_backend_parses_valid_response(monkeypatch):
    monkeypatch.setattr(validator, "VALIDATION_BACKEND", "llm")
    monkeypatch.setattr(
        validator,
        "_llm_call_blocking",
        lambda *a, **k: '{"valid": false, "errors": [{"rule": "net_sales_non_negative", "detail": "row 0"}]}',
    )

    df = pd.DataFrame({"net_sales": [-10]})
    result = await validate_business_rules(df)

    assert result["valid"] is False
    assert result["errors"][0]["rule"] == "net_sales_non_negative"
