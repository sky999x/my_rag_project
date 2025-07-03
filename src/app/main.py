import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager # Import asynccontextmanager for lifespan

# Absolutely import all core components from the top level of the src package
from src.data_processing.embedding import load_embedding_model
from src.vector_store.faiss_store import INDEX_SAVE_DIR, INDEX_NAME, METADATA_NAME
from src.retriever.faiss_retriever import FAISSRetriever
from src.generator.llm_generator import LLMGenerator # LLMGenerator handles internal model loading

# Define the request body model
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3 # By default, retrieves the 3 most relevant text blocks

# Define the response body model (optional, but recommended good practice)
class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_sources: List[Dict[str, str]]

# Global variables to store loaded components
# These components are loaded when the FastAPI application starts to avoid reloading for each request
embedding_model = None
retriever = None
generator = None

# Define the lifespan context manager function
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all RAG components when the FastAPI application starts up.
    This function also handles resource cleanup when the application shuts down.
    """
    global embedding_model, retriever, generator

    print("--- FastAPI Startup: Loading RAG components ---")

    # Construct the correct path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..')) # From app -> src -> my_rag_project

    # Absolute paths for index and metadata files
    absolute_index_save_dir = os.path.join(project_root, INDEX_SAVE_DIR)
    absolute_index_path = os.path.join(absolute_index_save_dir, INDEX_NAME)
    absolute_metadata_path = os.path.join(absolute_index_save_dir, METADATA_NAME)

    # 1. Load embedding model
    embedding_model = load_embedding_model()
    if embedding_model is None:
        raise RuntimeError("Failed to load embedding model during startup.")

    # 2. Initialize retriever
    try:
        retriever = FAISSRetriever(absolute_index_path, absolute_metadata_path, embedding_model)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize FAISSRetriever during startup: {e}")

    # 3. Initialize generator (LLM)
    generator = LLMGenerator() # LLMGenerator loads the model internally
    if generator.model is None or generator.tokenizer is None:
        raise RuntimeError("Failed to initialize LLMGenerator during startup.")

    print("--- FastAPI Startup: All RAG components loaded successfully! ---")

    # Yield control to the application to handle requests
    yield

    # --- Shutdown Logic (after yield) ---
    # You can add cleanup logic here if needed, e.g., closing database connections
    print("--- FastAPI Shutdown: Resources cleaned up (if any) ---")


# Pass the lifespan context manager to the FastAPI app
app = FastAPI(
    title="Modular RAG System Backend",
    description="A FastAPI backend for a Retrieval-Augmented Generation (RAG) system using FAISS and a local LLM.",
    lifespan=lifespan # Assign the lifespan context manager
)

@app.get("/")
async def read_root():
    """
    Root path for a simple health check.
    """
    return {"message": "Welcome to the Modular RAG System API! Use /query to ask questions."}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Receives a user query, executes the RAG process, and returns the generated answer.
    """
    # Check if RAG components are initialized (should be by lifespan)
    if retriever is None or generator is None:
        raise HTTPException(status_code=503, detail="RAG components not initialized. Please wait or check server logs.")

    query = request.query
    top_k = request.top_k

    print(f"\n--- Processing Query: '{query}' (top_k={top_k}) ---")

    # 1. Retrieval Phase
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)
    print(f"Retrieved {len(retrieved_chunks)} chunks.")

    # 2. Generation Phase
    # LLMGenerator's contexts parameter expects List[Dict[str, str]], which retrieved_chunks matches
    response = generator.generate_response(query, retrieved_chunks)
    print(f"Generated response: '{response[:100]}...'")

    # Prepare source information for the API response (only text and source, no internal metadata exposed)
    response_sources = []
    for chunk in retrieved_chunks:
        response_sources.append({
            "text": chunk.get('text', 'N/A'),
            "source": chunk['metadata'].get('source', 'N/A')
        })

    return QueryResponse(
        query=query,
        response=response,
        retrieved_sources=response_sources
    )