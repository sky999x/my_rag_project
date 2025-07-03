import os
from typing import List, Dict, Tuple
import faiss
import numpy as np
import json

# Absolute imports from the top level of the src package
# Ensure src and its subdirectories all contain __init__.py files
from src.data_processing.ingestion import load_documents_from_directory
from src.data_processing.chunking import split_documents_into_chunks
from src.data_processing.embedding import load_embedding_model, get_embeddings

# Define the save path for the vector index
# This path is relative to the project root directory
INDEX_SAVE_DIR = os.path.join("data", "processed")
INDEX_NAME = "my_rag_faiss_index.bin"
METADATA_NAME = "my_rag_faiss_index_metadata.json" # Used to save the original text chunks and their metadata

def create_and_save_faiss_index(chunks: List[Dict[str, str]],
                                 embedding_model) -> str:
    """
    Creates a FAISS index from text chunks and an embedding model,
    then saves both the index and the original text chunks to disk.

    Args:
        chunks (List[Dict[str, str]]): A list of text chunks, each containing 'text' and 'metadata'.
        embedding_model: The loaded SentenceTransformer embedding model.

    Returns:
        str: The full path to the saved FAISS index file, or None if creation/save fails.
    """
    if not chunks:
        print("No chunks provided to create index.")
        return None

    chunk_texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings for chunks to create FAISS index...")
    chunk_embeddings = get_embeddings(chunk_texts, embedding_model)

    if not chunk_embeddings:
        print("Failed to generate embeddings for chunks. Cannot create FAISS index.")
        return None

    embeddings_np = np.array(chunk_embeddings).astype('float32')
    embedding_dim = embeddings_np.shape[1]

    print(f"Creating FAISS index with dimension: {embedding_dim}...")
    index = faiss.IndexFlatL2(embedding_dim)

    print(f"Adding {len(embeddings_np)} embeddings to FAISS index...")
    index.add(embeddings_np)
    print("Embeddings added to FAISS index.")

    # Ensure the INDEX_SAVE_DIR path exists, which should be relative to the project root
    # Get the directory of the current file (faiss_store.py)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from current_file_dir (vector_store -> src -> my_rag_project) to the project root
    project_root_for_save = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    absolute_index_save_dir = os.path.join(project_root_for_save, INDEX_SAVE_DIR)

    os.makedirs(absolute_index_save_dir, exist_ok=True)

    index_path = os.path.join(absolute_index_save_dir, INDEX_NAME)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")

    metadata_path = os.path.join(absolute_index_save_dir, METADATA_NAME)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4) # ensure_ascii=False allows non-ASCII characters if needed
    print(f"Chunks metadata saved to: {metadata_path}")

    return index_path

def load_faiss_index(index_path: str):
    """
    Loads a FAISS index from disk.

    Args:
        index_path (str): The full path to the FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index instance.
    """
    if not os.path.exists(index_path):
        print(f"FAISS index file not found at: {index_path}")
        return None

    print(f"Loading FAISS index from: {index_path}...")
    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully.")
    return index

def load_faiss_index_metadata(metadata_path: str) -> List[Dict[str, str]]:
    """
    Loads the original text chunks and their metadata corresponding to a FAISS index from disk.

    Args:
        metadata_path (str): The full path to the JSON file storing metadata.

    Returns:
        List[Dict[str, str]]: A list of original text chunks and their metadata.
    """
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        return []

    print(f"Loading FAISS index metadata from: {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print("Metadata loaded successfully.")
    return metadata


if __name__ == "__main__":
    # This is a simple test area to verify the function's functionality

    # Construct raw data path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from current_file_dir (vector_store -> src -> my_rag_project)
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    raw_data_path = os.path.join(project_root, 'data', 'raw')

    # Ensure INDEX_SAVE_DIR path is relative to the project root
    absolute_index_save_dir = os.path.join(project_root, INDEX_SAVE_DIR)
    absolute_index_path = os.path.join(absolute_index_save_dir, INDEX_NAME)
    absolute_metadata_path = os.path.join(absolute_index_save_dir, METADATA_NAME)

    print(f"Using raw data path: {raw_data_path}")
    print(f"Index will be saved/loaded from: {absolute_index_path}")

    # 1. Load documents (using Task 2.1)
    print("\n--- Step 1: Loading documents ---")
    loaded_docs = load_documents_from_directory(raw_data_path)

    if loaded_docs:
        # 2. Split documents into chunks (using Task 2.2)
        print("\n--- Step 2: Splitting documents into chunks ---")
        chunks = split_documents_into_chunks(loaded_docs, chunk_size=500, chunk_overlap=100)

        if chunks:
            # 3. Load embedding model (using Task 3.0)
            print("\n--- Step 3: Loading embedding model ---")
            embedding_model = load_embedding_model()

            if embedding_model:
                # 4. Create and save FAISS index
                print("\n--- Step 4: Creating and saving FAISS index ---")
                created_index_path = create_and_save_faiss_index(chunks, embedding_model)

                if created_index_path:
                    print(f"\nFAISS index creation and save successful: {created_index_path}")

                    # 5. Attempt to load index and metadata (verify successful save)
                    print("\n--- Step 5: Loading FAISS index and metadata for verification ---")
                    loaded_index = load_faiss_index(absolute_index_path)
                    loaded_metadata = load_faiss_index_metadata(absolute_metadata_path)

                    if loaded_index and loaded_metadata:
                        print(f"Loaded index has {loaded_index.ntotal} vectors.")
                        print(f"Loaded metadata has {len(loaded_metadata)} chunks.")
                        print("\nVerification successful: Index and metadata can be loaded back.")
                    else:
                        print("Verification failed: Could not load index or metadata.")
                else:
                    print("Failed to create FAISS index.")
            else:
                print("Embedding model failed to load. Cannot proceed with FAISS index creation.")
        else:
            print("No chunks generated. Cannot create FAISS index.")
    else:
        print("No documents loaded. Cannot create FAISS index.")
