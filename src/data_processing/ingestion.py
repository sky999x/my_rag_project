import os
from typing import List, Dict
from pypdf import PdfReader
import re # Import the regular expression module

def clean_text_for_embedding(text: str) -> str:
    """
    Cleans text extracted from documents, removing or replacing special Unicode characters
    that might cause issues for embedding models.
    Primarily targets non-standard characters that might arise during pypdf extraction,
    such as surrogate pairs or private use area characters.
    """
    if not isinstance(text, str): # Ensure input is a string
        return ""

    # Remove Unicode surrogate pairs (U+D800 to U+DFFF)
    # These are special sequences in UTF-16 encoding used to represent characters
    # outside the Basic Multilingual Plane (BMP). When they appear unpaired or
    # incorrectly, they usually indicate encoding errors or invalid characters.
    text = re.sub(r'[\ud800-\udfff]', '', text)

    # Remove Private Use Area (PUA) characters (e.g., Wingdings/Webdings might map here)
    # Common PUA-A range is U+E000 to U+F8FF
    text = re.sub(r'[\ue000-\uf8ff]', '', text)

    # Remove or replace other special Unicode characters that might cause issues
    # For example: special dashes, ellipses, invisible zero-width spaces, etc.
    text = text.replace('\u2013', '-').replace('\u2014', '--') # Replace EN DASH and EM DASH
    text = text.replace('\u2022', '*') # Replace bullet point
    text = text.replace('\u2026', '...') # Replace ellipsis
    text = text.replace('\u00a0', ' ') # Replace non-breaking space with regular space
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u205f\u3000]', '', text) # Remove zero-width spaces, line/paragraph separators, etc.

    # Replace multiple consecutive whitespace characters (spaces, tabs, newlines) with a single space
    # And remove leading/trailing whitespace from the string
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove excessive blank lines (ensure only single blank lines remain)
    # Re-check to ensure no accidental blank lines due to multiple replacements
    text = re.sub(r'(\n\s*){2,}', '\n\n', text) # Keep single blank lines, remove excessive ones

    return text

def load_documents_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Loads all .txt and .pdf files from the specified directory, extracts their text content, and cleans them.

    Args:
        directory_path (str): The path to the directory containing raw documents.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'text' (document content) and 'source' (file path).
    """
    documents = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            if filename.endswith(".txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Clean the text from TXT files as well, just in case
                        cleaned_text = clean_text_for_embedding(text)
                        if cleaned_text: # Only add if content remains after cleaning
                            documents.append({"text": cleaned_text, "source": file_path})
                            print(f"Loaded TXT file: {filename}")
                        else:
                            print(f"Warning: TXT file {filename} resulted in empty text after cleaning. Skipping.")
                except Exception as e:
                    print(f"Error loading TXT file {filename}: {e}")
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    text_pages = []
                    for page in reader.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text_pages.append(extracted_text)

                    full_text = "\n".join(text_pages)

                    # !!! Apply the cleaning function !!!
                    cleaned_full_text = clean_text_for_embedding(full_text)

                    # Skip if text becomes empty after cleaning
                    if not cleaned_full_text:
                        print(f"Warning: PDF {filename} resulted in empty text after cleaning. Skipping.")
                        continue

                    documents.append({"text": cleaned_full_text, "source": file_path})
                    print(f"Loaded PDF file: {filename}")
                except Exception as e:
                    print(f"Error loading PDF file {filename}: {e}")
            else:
                print(f"Skipping unsupported file type: {filename}")
    return documents

if __name__ == "__main__":
    # This is a simple test area to verify the function's functionality
    # Construct the correct relative path
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    raw_data_path = os.path.join(project_root, 'data', 'raw')

    print(f"Attempting to load documents from: {raw_data_path}")
    loaded_docs = load_documents_from_directory(raw_data_path)

    if loaded_docs:
        print(f"\nSuccessfully loaded {len(loaded_docs)} documents.")
        for i, doc in enumerate(loaded_docs):
            print(f"--- Document {i+1} ---")
            print(f"Source: {doc['source']}")
            # Print the first 200 characters as a preview, handling encoding issues safely
            preview_text = doc['text'][:200]
            try:
                print(f"Content Preview: {preview_text.encode('utf-8', 'ignore').decode('utf-8')}...")
            except Exception as e:
                print(f"Content Preview (encoding issue): {e} - Showing raw first 50 chars: {doc['text'][:50]}...")
            print("-" * 20)
    else:
        print("No documents loaded. Please check the directory path and file types.")
