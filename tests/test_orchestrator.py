import pandas as pd
import pytest

from app.agents.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_clean_dataset_returns_success(monkeypatch):
    orch = Orchestrator()

    async def fake_execute(_query):
        data = pd.DataFrame(
            {
                "net_sales": [100, 200],
                "discount": [10, 20],
                "transaction_date": ["2020-01-01", "2021-01-01"],
            }
        )
        return {"success": True, "data": data, "metadata": {"source": "test"}}

    monkeypatch.setattr(orch.query_agent, "execute", fake_execute)

    result = await orch.process_request("ventas del año pasado")
    assert result["status"] == "success"
    assert all(r["passed"] for r in result["validation"])


@pytest.mark.asyncio
async def test_illogical_dataset_flagged_suspect(monkeypatch):
    orch = Orchestrator()

    async def fake_execute(_query):
        data = pd.DataFrame({"net_sales": [-50, 100]})
        return {"success": True, "data": data, "metadata": {"source": "test"}}

    monkeypatch.setattr(orch.query_agent, "execute", fake_execute)

    result = await orch.process_request("ventas netas")
    assert result["status"] == "suspect"
    assert any(not r["passed"] for r in result["validation"])


@pytest.mark.asyncio
async def test_query_failure_short_circuits(monkeypatch):
    orch = Orchestrator()

    async def fake_execute(_query):
        return {"success": False}

    monkeypatch.setattr(orch.query_agent, "execute", fake_execute)

    result = await orch.process_request("algo")
    assert "error" in result
