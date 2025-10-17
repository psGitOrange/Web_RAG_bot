from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from rag.models import Base, IngestionRecord
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# async def init_db():
#     """Initialize database tables"""
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session  # Auto-closes after exiting

async def create_ingestion_record(urls: list[str], status: str = 'completed',
        chunks_count: int = None, error_message: str = None, metadata: dict = None):
    """Create a new ingestion record"""
    async with async_session() as session:
        record = IngestionRecord(
            urls=urls,
            status=status,
            # chunks_count=chunks_count,
            error_message=error_message,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

async def get_all_records(limit: int = 10):
    """Get all ingestion records"""
    async with async_session() as session:
        result = await session.execute(
            select(IngestionRecord)
            .order_by(IngestionRecord.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

async def get_records_by_status(status: str):
    """Get records by status"""
    async with async_session() as session:
        result = await session.execute(
            select(IngestionRecord).where(IngestionRecord.status == status)
        )
        return result.scalars().all()