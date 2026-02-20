"""Shared test fixtures and async test client setup."""

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import create_app
from app.models.database import Base, engine

SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.fixture(scope="session")
def app():
    """Create a fresh application instance for the test session."""
    return create_app()


@pytest.fixture
async def client(app):
    """Async HTTP test client — no real server needed."""
    # Reset DB tables before each test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Valid API key headers."""
    return {"X-API-Key": settings.API_KEY}


@pytest.fixture
def sample_transcript():
    """A simple test transcript."""
    return (
        "Agent: Good morning, thank you for calling Jio support. How can I help you?\n"
        "Customer: Hi, my internet is slow since yesterday.\n"
        "Agent: I'm sorry to hear that. Let me check your connection.\n"
        "Customer: Okay, thanks.\n"
        "Agent: I can see there was a brief outage in your area. It should be resolved now. "
        "Could you restart your router?\n"
        "Customer: Sure, let me try... Yes, it's working now! Thank you.\n"
        "Agent: Great! Is there anything else I can help with?\n"
        "Customer: No, that's all. Thanks!\n"
        "Agent: Thank you for calling. Have a great day!"
    )


@pytest.fixture
def sample_config_payload():
    """Payload for creating a test configuration."""
    return {
        "client_id": "test_telecom",
        "client_name": "Test Telecom Provider",
        "domain": "telecom",
        "description": "Test configuration",
        "products": ["prepaid", "postpaid"],
        "compliance_policies": [
            {
                "id": "TEST-001",
                "name": "Test Policy",
                "description": "A test compliance policy",
                "severity": "high",
            }
        ],
        "risk_triggers": ["cancel", "complaint"],
        "quality_criteria": {
            "greeting_required": True,
            "closing_required": True,
            "empathy_keywords": ["sorry", "understand"],
            "resolution_confirmation_required": True,
            "prohibited_phrases": ["calm down"],
        },
        "call_outcome_categories": ["resolved", "escalated"],
    }


def load_sample(name: str) -> dict:
    """Load a sample conversation from the samples directory."""
    path = SAMPLES_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
