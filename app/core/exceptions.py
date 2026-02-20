"""Custom exceptions for the application.

Each exception maps to a specific HTTP error response with
a machine-readable error code.
"""

from fastapi import HTTPException, status


class AppException(HTTPException):
    """Base application exception with error code support."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(status_code=status_code, detail=message)


class ConfigurationNotFoundError(AppException):
    """Raised when a client configuration is not found."""

    def __init__(self, config_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="CONFIGURATION_NOT_FOUND",
            message=f"Configuration '{config_id}' not found",
            details={"config_id": config_id},
        )


class ConfigurationAlreadyExistsError(AppException):
    """Raised when trying to create a config that already exists."""

    def __init__(self, config_id: str):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            code="CONFIGURATION_ALREADY_EXISTS",
            message=f"Configuration '{config_id}' already exists",
            details={"config_id": config_id},
        )


class InvalidInputError(AppException):
    """Raised when request input is invalid."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVALID_INPUT",
            message=message,
            details=details,
        )


class AudioProcessingError(AppException):
    """Raised when audio file processing fails."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="AUDIO_PROCESSING_ERROR",
            message=message,
            details=details,
        )


class FileTooLargeError(AppException):
    """Raised when uploaded file exceeds size limit."""

    def __init__(self, max_size_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            code="FILE_TOO_LARGE",
            message=f"File exceeds maximum size of {max_size_mb}MB",
            details={"max_size_mb": max_size_mb},
        )


class UnsupportedFileTypeError(AppException):
    """Raised when uploaded file type is not supported."""

    def __init__(self, content_type: str, allowed_types: list[str]):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            code="UNSUPPORTED_FILE_TYPE",
            message=f"File type '{content_type}' is not supported",
            details={"content_type": content_type, "allowed_types": allowed_types},
        )


class AIEngineError(AppException):
    """Raised when the AI engine fails to process a request."""

    def __init__(
        self, message: str = "AI analysis failed", details: dict | None = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AI_ENGINE_ERROR",
            message=message,
            details=details,
        )
