Modular RAG System Backend with FastAPI and LLMs
This project implements a robust, end-to-end Retrieval-Augmented Generation (RAG) system backend designed for intelligent Q&A from custom, unstructured documents (e.g., PDFs, TXTs). Built with Python and FastAPI, it provides a high-performance, scalable, and modular API for contextual information retrieval and LLM-powered response generation.

Key Features
End-to-End RAG Pipeline: Covers document ingestion, text chunking, embedding generation, vector storage, retrieval, and LLM-based response generation.

High-Performance FastAPI Backend: Exposes a clean, asynchronous RESTful API for efficient query processing.

Modular and Scalable Architecture: Designed with distinct components (data processing, vector store, retriever, generator) for easy maintenance, upgrades, and horizontal scaling.

Efficient Vector Database: Utilizes FAISS for fast and accurate similarity search over custom knowledge bases.

Robust Data Preprocessing: Includes advanced text cleaning to handle complex Unicode characters and ensure high-quality embeddings from diverse document sources (PDF, TXT).

Local LLM Integration: Seamlessly integrates with open-source Large Language Models (e.g., Qwen1.5-0.5B-Chat) for privacy-preserving and cost-effective solutions.

Interactive API Documentation: Automatically generated Swagger UI for easy testing and integration with frontend applications.

Technology Stack
Backend Framework: FastAPI (Python)

LLM Orchestration: Custom RAG pattern implementation

Vector Database: FAISS

Embedding Model: BAAI/bge-small-zh-v1.5 (Sentence Transformers)

Large Language Model (LLM): Qwen/Qwen1.5-0.5B-Chat (Hugging Face Transformers)

Document Parsing: pypdf for PDF extraction

Data Processing: NumPy, custom Python scripts

Dependency Management: pip

Setup and Local Development
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.9+

pip (Python package installer)

Git

1. Clone the Repository
First, clone this repository to your local machine:

git clone https://github.com/YourUsername/your-repository-name.git
cd your-repository-name

(Replace YourUsername and your-repository-name with your actual GitHub username and repository name.)

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies:

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
Install all required Python packages:

pip install -r requirements.txt

(If requirements.txt is missing, you can generate one by running pip freeze > requirements.txt after manually installing all dependencies like fastapi, uvicorn, pypdf, faiss-cpu, numpy, sentence-transformers, transformers, torch.)

4. Prepare Your Data
Place your raw PDF and TXT documents into the data/raw/ directory within the project. For demonstration purposes, you can include sample English documents here.

5. Process Data and Build FAISS Index
Run the data processing script to load documents, split them into chunks, generate embeddings, and build/save the FAISS vector index along with its metadata.

python src/data/process_data.py

(You should see messages indicating successful loading, chunking, embedding, and FAISS index creation/save.)

6. Run the FastAPI Application
Start the FastAPI backend server:

uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload

(The application will be running at http://127.0.0.1:8000. The --reload flag enables auto-reloading on code changes during development.)

7. Access API Documentation (Swagger UI)
Once the server is running, open your web browser and navigate to:
http://127.0.0.1:8000/docs

Here, you can interact with the API, test the /query endpoint, and view the request/response models.

Example API Usage
You can use the /query endpoint via the Swagger UI or a tool like Postman/curl to send a POST request.

Endpoint: POST /query
Request Body (JSON):

{
  "query": "What is artificial intelligence?",
  "top_k": 3
}

Example Response (JSON):

{
  "query": "What is artificial intelligence?",
  "response": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
  "retrieved_sources": [
    {
      "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
      "source": "wiki.pdf"
    }
  ]
}

Contact
For any questions or further discussion, feel free to reach out.

aiconsult.engineer
aiconsult.engineer@protonmail.com
