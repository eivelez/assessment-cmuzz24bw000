from typing import Dict, Any
from app.agents.query_agent import QueryAgent
from app.utils.logger import logger
from app.utils.dispatcher import dispatcher
# Imported for its registration side effect: importing the module runs the
# @dispatcher.register_validator decorator so the validator is available here.
import app.agents.validator  # noqa: F401

class Orchestrator:
    def __init__(self):
        self.query_agent = QueryAgent()

    async def process_request(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Processing user query: {user_query}")

        # 1. Generate and execute SQL
        query_result = await self.query_agent.execute(user_query)

        if not query_result.get("success"):
            return {"error": "Failed to generate valid results"}

        dataset = query_result.get("data")

        # 2. Validate coherence before returning to the UI.
        validation = await dispatcher.run_validators(dataset)
        passed = all(r.get("passed") for r in validation)
        degraded = any(r.get("degraded") for r in validation)
        status = "success" if (passed and not degraded) else "suspect"

        if status == "suspect":
            # Out of scope to surface in the Chainlit UI for now — log it and
            # mark the result as suspect at the API level.
            logger.warning("Dataset flagged as suspect: %s", validation)

        return {
            "status": status,
            "data": dataset,
            "metadata": query_result.get("metadata"),
            "validation": validation,
        }
