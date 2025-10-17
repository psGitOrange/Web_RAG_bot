import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.indexing import create_chroma_index
from rag.querying import create_query_engine
from rag.chat_engine import create_chat_engine
from rag.model_crud import create_ingestion_record, get_all_records

app = FastAPI()


@app.get("/")
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "message": "Welcome to the RAG Query API!, with access to webpage",
        "endpoints": {
            "/ingest-url": "POST - ingest webpage url to vector store for factual answers",
            "/query": "POST - Retrieve query response",
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

        record = await create_ingestion_record(
            urls=req.urls,
            status='completed',
        )

        return {
            "status": "success",
            "message": f"Successfully indexed {len(req.urls)} URLs",
            "record_id": str(record.id),
            "urls": req.urls
        }
    except Exception as e:
        await create_ingestion_record(
            urls=req.urls,
            status='failed',
            error_message=str(e)
        )

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
