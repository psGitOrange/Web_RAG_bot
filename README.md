# Web_RAG_bot
RAG bot with capabilities to retrieve webpages text to give factual answers using llama-index

### Start FastAPI
```bash
   uvicorn app:app --port 8000 --reload
   ```
   Endpoints:
   * `POST /ingest-url` – Ingest web page data into vector stores.
   * `POST /query` – Retrieve query response.
   

### Ingest URLs
curl -X POST http://localhost:8000/ingest-url \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com", "https://example.org"]}'

### Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": ["what is rag"]}'
