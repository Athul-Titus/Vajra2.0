"""Database models and async session management.

Uses SQLAlchemy 2.0 async with aiosqlite for zero-infrastructure
persistence of client configurations.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, Text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class ClientConfigurationModel(Base):
    """Persisted client configuration.

    Stores the full JSON configuration that drives analysis behavior
    for each client.
    """

    __tablename__ = "client_configurations"

    client_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    client_name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(50), nullable=False, default="telecom")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    config_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<ClientConfig {self.client_id}>"


# --- Async engine and session factory ---

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.is_development,
    future=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncSession:
    """Dependency that provides an async database session.

    Usage in FastAPI endpoints:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database() -> None:
    """Create all database tables on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Dispose engine connections on application shutdown."""
    await engine.dispose()
