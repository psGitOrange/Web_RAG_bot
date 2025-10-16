import os
import asyncio
import tempfile
from typing import Literal, Optional

# import edge_tts
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.indexing import create_chroma_index
from rag.querying import create_query_engine
from rag.chat_engine import create_chat_engine

app = FastAPI()

@app.get("/")
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "message": "Welcome to the RAG Query API!, Edge TTS API is running",
        "available_voices": {
            "en": "English (US) - Female (Ava)",
            "hi": "Hindi - Female (Swara)"
        },
        "endpoints": {
            "/query": "POST - Retrieve query response",
            "/tts": "POST - Generate speech from text"
        }
    }

index = create_chroma_index()
query_engine = create_query_engine(index)
chat_engine = create_chat_engine(index)

class IndexRequest(BaseModel):
    urls: list[str]

@app.post("/ingest-url")
async def index_web_page(req: IndexRequest):
    try:
        index = create_chroma_index(web_urls=req.urls)
        chat_engine = create_chat_engine(index)

        return {
            "status": "success",
            "message": f"Successfully indexed {len(req.urls)} URLs",
            "urls": req.urls
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def ask_question(req: QueryRequest):
    # response = query_engine.query(req.query)
    response = chat_engine.chat(req.query)
    return {"response": response.response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
