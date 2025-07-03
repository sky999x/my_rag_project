import os
from typing import List, Dict, Tuple
import faiss
import numpy as np

# Absolute imports from the top level of the src package
# Ensure src and its subdirectories all contain __init__.py files
from src.data_processing.embedding import load_embedding_model, get_embeddings
from src.vector_store.faiss_store import load_faiss_index, load_faiss_index_metadata, INDEX_SAVE_DIR, INDEX_NAME, METADATA_NAME

class FAISSRetriever:
    """
    Retriever based on FAISS vector database and SentenceTransformer embedding model.
    """
    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 embedding_model):
        """
        Initializes the retriever.

        Args:
            index_path (str): The full path to the FAISS index file.
            metadata_path (str): The full path to the JSON file storing raw text chunk metadata.
            embedding_model: The loaded SentenceTransformer embedding model.
        """
        self.index = load_faiss_index(index_path)
        self.metadata = load_faiss_index_metadata(metadata_path)
        self.embedding_model = embedding_model

        if self.index is None or not self.metadata:
            raise ValueError("Failed to load FAISS index or metadata. Retriever cannot be initialized.")
        if self.embedding_model is None:
            raise ValueError("Embedding model is not loaded. Retriever cannot be initialized.")

        print(f"FAISSRetriever initialized with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieves the most relevant text chunks based on the query text.

        Args:
            query (str): The user's input query string.
            top_k (int): The number of most relevant text chunks to return.

        Returns:
            List[Dict[str, str]]: A list of the most relevant text chunks,
                                  each dictionary containing 'text' and 'metadata'.
        """
        if not query or not query.strip():
            print("Query is empty or whitespace-only. Cannot perform retrieval.")
            return []

        # 1. Convert the query text into an embedding vector
        query_embedding_list = get_embeddings([query], self.embedding_model)

        if not query_embedding_list:
            print("Failed to generate embedding for the query. Cannot perform retrieval.")
            return []

        query_embedding = np.array(query_embedding_list[0]).astype('float32').reshape(1, -1) # reshape for FAISS

        # 2. Search for the most similar vectors in the FAISS index
        # D: List of distances, I: List of indices
        # FAISS returns L2 distance, smaller value indicates higher similarity
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_chunks = []
        # 3. Retrieve the corresponding original text chunks and metadata based on indices
        for i, idx in enumerate(indices[0]): # indices[0] because there's only one query
            if 0 <= idx < len(self.metadata): # Ensure index is valid
                chunk = self.metadata[idx]
                # Optionally, add the distance to the metadata
                chunk['distance'] = float(distances[0][i])
                retrieved_chunks.append(chunk)
            else:
                print(f"Warning: Retrieved invalid index {idx}. Skipping.")

        print(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:50]}...'")
        return retrieved_chunks

# Test area
if __name__ == "__main__":
    # Construct the correct path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..')) # From retriever -> src -> my_rag_project

    # Absolute paths for index and metadata files
    absolute_index_save_dir = os.path.join(project_root, INDEX_SAVE_DIR) # Path imported from faiss_store
    absolute_index_path = os.path.join(absolute_index_save_dir, INDEX_NAME)
    absolute_metadata_path = os.path.join(absolute_index_save_dir, METADATA_NAME)

    print(f"Attempting to load FAISS index from: {absolute_index_path}")
    print(f"Attempting to load metadata from: {absolute_metadata_path}")

    # 1. Load embedding model
    print("\n--- Step 1: Loading embedding model for Retriever ---")
    embedding_model = load_embedding_model()

    if embedding_model:
        # 2. Initialize FAISSRetriever
        print("\n--- Step 2: Initializing FAISSRetriever ---")
        try:
            retriever = FAISSRetriever(absolute_index_path, absolute_metadata_path, embedding_model)
            print("FAISSRetriever initialized successfully.")

            # 3. Perform retrieval test
            print("\n--- Step 3: Performing Retrieval Test ---")
            test_query = "What is the project about?" # A query about the Upwork project
            retrieved_results = retriever.retrieve(test_query, top_k=2) # Retrieve 2 most relevant

            if retrieved_results:
                print(f"\nSuccessfully retrieved {len(retrieved_results)} results for query: '{test_query}'")
                for i, chunk in enumerate(retrieved_results):
                    print(f"\n--- Retrieved Chunk {i+1} (Distance: {chunk.get('distance', 'N/A'):.4f}) ---")
                    print(f"Source: {chunk['metadata'].get('source', 'N/A')}")
                    # Print the first 200 characters as a preview
                    print(f"Content Preview: {chunk['text'][:200]}...")
            else:
                print("No results retrieved. Please check if your FAISS index and metadata are correctly populated.")

            # Another query, for example, about the content of a test.txt file
            test_query_2 = "What is the purpose of the test document?"
            retrieved_results_2 = retriever.retrieve(test_query_2, top_k=1)
            if retrieved_results_2:
                print(f"\nSuccessfully retrieved {len(retrieved_results_2)} results for query: '{test_query_2}'")
                for i, chunk in enumerate(retrieved_results_2):
                    print(f"\n--- Retrieved Chunk {i+1} (Distance: {chunk.get('distance', 'N/A'):.4f}) ---")
                    print(f"Source: {chunk['metadata'].get('source', 'N/A')}")
                    print(f"Content Preview: {chunk['text'][:200]}...")

        except ValueError as ve:
            print(f"Retriever initialization failed: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during retrieval test: {e}")
    else:
        print("Embedding model failed to load. Cannot test retriever.")
