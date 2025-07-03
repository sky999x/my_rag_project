import os
from sentence_transformers import SentenceTransformer
from typing import List, Union # Import Union for more precise type checking

# This is a very commonly used and high-performing Chinese/English mixed embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5" # Suitable for both Chinese and English mixed languages

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Loads a pre-trained SentenceTransformer embedding model.
    The model files will be downloaded and cached locally.

    Args:
        model_name (str): The model name from the Hugging Face model hub.

    Returns:
        SentenceTransformer: The loaded embedding model instance.
    """
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading embedding model {model_name}: {e}")
        print("Please check your internet connection and model name.")
        return None

def get_embeddings(texts: List[str], model: SentenceTransformer) -> List[List[float]]:
    """
    Converts a list of texts into their corresponding embedding vectors.
    Performs strict text validation and filtering before encoding.

    Args:
        texts (List[str]): A list of text strings to be embedded.
        model (SentenceTransformer): The loaded embedding model instance.

    Returns:
        List[List[float]]: A list of embedding vectors for the texts.
    """
    if model is None:
        print("Embedding model is not loaded. Cannot get embeddings.")
        return []

    valid_texts_to_encode = []
    original_indices_map = []

    print(f"--- get_embeddings: Starting input validation for {len(texts)} texts ---")
    for i, text in enumerate(texts):
        # 1. Check type: Must be a string
        if not isinstance(text, str):
            print(f"Error: Invalid type found at original index {i}. Expected string, got {type(text)}. Skipping.")
            continue

        # 2. Check for empty or whitespace-only
        cleaned_text = text.strip()
        if not cleaned_text:
            print(f"Warning: Skipping empty or whitespace-only text at original index {i}. Original length: {len(text)}.")
            # Can print the repr of the original text to check for hidden characters
            # print(f"Raw content (repr): {repr(text)}") # Keep this commented for production
            continue

        # 3. Additional check: Ensure string is encodable Unicode (rare but possible)
        try:
            # Attempt to encode and then decode; this can filter out some unrepresentable characters.
            # The purpose here is to check for internal encoding issues.
            test_encode = cleaned_text.encode('utf-8', 'ignore').decode('utf-8')
            # Further check if the cleaned text still has content
            if not test_encode.strip():
                print(f"Warning: Text at index {i} became empty after robust encoding/decoding. Skipping.")
                continue

        except UnicodeEncodeError as e:
            print(f"Error: UnicodeEncodeError for text at original index {i}. Content might contain problematic characters. Skipping. Error: {e}")
            print(f"Problematic text (first 100 chars): {cleaned_text[:100]}...")
            continue
        except Exception as e:
            print(f"Unexpected error during text encoding check for index {i}: {e}. Skipping.")
            continue

        valid_texts_to_encode.append(cleaned_text) # Use the cleaned text for encoding
        original_indices_map.append(i)

    print(f"--- get_embeddings: Validation complete. {len(valid_texts_to_encode)} texts are valid. ---")

    if not valid_texts_to_encode:
        print("No valid texts found for embedding after strict filtering.")
        return []

    print(f"Generating embeddings for {len(valid_texts_to_encode)} valid texts...")
    try:
        # Attempt encoding; if it fails again, this try-except block will catch it
        valid_embeddings = model.encode(valid_texts_to_encode, normalize_embeddings=True).tolist()
        print("Embeddings generated successfully.")
    except TypeError as e:
        print(f"Critical Error during model.encode(): {e}")
        print("This usually means the text inputs are not in the expected format (e.g., non-string elements).")
        print("Examining problematic texts (first 5 of valid_texts_to_encode for sanity check):")
        for i, txt in enumerate(valid_texts_to_encode[:min(5, len(valid_texts_to_encode))]): # Ensure no out-of-bounds indexing
            print(f"  Valid Text {i+1} (Type: {type(txt)}, Len: {len(txt)}): {repr(txt[:100])}...")
        print("If you see this, the issue is very subtle and might require deep library debugging or data inspection.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during model.encode(): {e}")
        return []

    # Construct the final embeddings list, consistent with the original input list's length and order
    final_ordered_embeddings = [None] * len(texts) # Initialize a list with the same length as original texts
    for i, original_idx in enumerate(original_indices_map):
        final_ordered_embeddings[original_idx] = valid_embeddings[i]

    # Finally, return embeddings for all valid texts, removing any skipped None values
    return [emb for emb in final_ordered_embeddings if emb is not None]


if __name__ == "__main__":
    # This is a simple test area to verify the function's functionality
    # It will only run when embedding.py is executed as a script.

    # 1. Load the embedding model
    embedding_model = load_embedding_model()

    if embedding_model:
        # 2. Prepare some test texts, including some that might cause issues (empty or whitespace-only)
        test_texts = [
            "This is a sentence about artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
            "Cats love to play.",
            "", # Simulate empty text
            "   \n\t", # Simulate text with only whitespace characters
            "Dogs love to run.",
        ]

        # 3. Get embedding vectors for these texts
        embeddings = get_embeddings(test_texts, embedding_model)

        if embeddings:
            print(f"\nGenerated {len(embeddings)} embeddings (after filtering invalid inputs).")
            # Ensure the embeddings list is not empty in case all test texts were filtered out
            if embeddings:
                print(f"Shape of first embedding: {len(embeddings[0])} dimensions.")
                print("\n--- First Embedding Preview (First 5 dimensions) ---")
                print(embeddings[0][:5])
                print("-" * 30)
        else:
            print("No valid embeddings generated from test texts.")
    else:
        print("Embedding model failed to load. Cannot proceed with embedding test.")

