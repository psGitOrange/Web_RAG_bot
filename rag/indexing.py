from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
import chromadb
import os


def create_chroma_index(docs_dir="./data", persist_dir="./indexes/chroma",
                        collection_name="web_docs", web_urls=None):
    """Create and persist a Chroma index from documents."""

    # Handle new documents being added to existing index
    if os.path.exists(persist_dir) and web_urls:
        print(f"Adding new documents to existing index at {persist_dir}")
        index = load_chroma_index(persist_dir, collection_name)

        # Add new documents to existing index
        new_documents = SimpleWebPageReader(html_to_text=True).load_data(web_urls)
        doc_nodes = Settings.node_parser.get_nodes_from_documents(new_documents)
        for node in doc_nodes:
            index.insert_nodes([node])

        print(f"Added {len(new_documents)} new documents to index")
        return index

    # Check if index already exists
    if os.path.exists(persist_dir):
        print(f"Loading existing index from {persist_dir}")
        return load_chroma_index(persist_dir, collection_name)

    # embed model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Load documents
    documents = SimpleDirectoryReader(docs_dir).load_data()
    doc_nodes = Settings.node_parser.get_nodes_from_documents(documents)

    # Create Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.create_collection(collection_name)

    # Create vector store and index
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        doc_nodes, embed_model=embed_model,
        storage_context=storage_context,
    )

    print(f"Created and persisted index with {len(documents)} documents")
    return index


def load_chroma_index(persist_dir="./indexes/chroma", collection_name="web_docs"):
    """Load a persisted Chroma index."""
    # embed model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    return index