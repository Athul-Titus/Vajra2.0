"""Request schemas for conversation analysis.

Defines how clients submit conversations for analysis —
either as a text transcript or an audio file upload.
"""

from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    """JSON request body for analyzing a text transcript.

    Used when the client sends a text conversation (not a file upload).
    """

    transcript: str = Field(
        ...,
        min_length=10,
        max_length=50_000,
        description="The conversation transcript to analyze",
        examples=[
            "Agent: Good morning, thank you for calling Jio support.\n"
            "Customer: Hi, my internet is very slow today.\n"
            "Agent: I'm sorry to hear that. Let me check your account."
        ],
    )
    config_id: str = Field(
        "telecom_default",
        description="Client configuration ID to use for analysis",
        examples=["telecom_default", "jio_telecom"],
        pattern=r"^[a-z0-9_]+$",
    )


# Note: Audio analysis uses multipart/form-data, so the request
# is handled via FastAPI's File() and Form() parameters directly
# in the endpoint, not via a Pydantic model.
