from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


def create_query_engine(index, similarity_top_k=3):
    """Create a query engine from the index."""
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
    )

    return query_engine