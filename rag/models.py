from sqlalchemy import Column, String, Integer, DateTime, Text, ARRAY, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
# from datetime import datetime, UTC
from sqlalchemy.sql import func

Base = declarative_base()


class IngestionRecord(Base):
    __tablename__ = "ingestion_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    urls = Column(ARRAY(Text), nullable=False)  # unique=True
    status = Column(String(20), nullable=False, default='completed')  # completed, failed
    created_at = Column(DateTime, server_default=func.now())
    chunks_count = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)  # Error tracking (if ingestion fails)

