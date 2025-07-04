import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Define the LLM model name to be used
# Qwen1.5-0.5B-Chat is a relatively small model, suitable for initial testing and local execution.
# For production environments or higher quality responses, you might need a larger model (e.g., Qwen1.5-7B-Chat)
# or use cloud services like OpenAI (which requires an API Key).
LLM_MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

class LLMGenerator:
    """
    Text generator based on a pre-trained Large Language Model (LLM).
    """
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """
        Initializes the generator, loading the language model and tokenizer.
        """
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading LLM model: {model_name} on device: {self.device}...")
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load model
            # trust_remote_code=True is needed for some models with custom code.
            # torch_dtype=torch.bfloat16 (or torch.float16) can save GPU memory but requires a compatible GPU.
            # If memory is insufficient, you can try load_in_8bit=True or load_in_4bit=True (requires bitsandbytes library).
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, # Use float16 for GPU to save memory, float32 for CPU
                device_map="auto" if self.device == "cuda" else None, # Automatically assign to GPU, None for CPU
                trust_remote_code=True
            )

            # Ensure the model is on the correct device
            if self.device == "cpu" and self.model: # If CPU, ensure model is on CPU
                self.model.to("cpu")
            elif self.device == "cuda" and self.model:
                self.model.to(self.device)

            self.model.eval() # Set to evaluation mode, no gradient calculation
            print("LLM model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading LLM model {model_name}: {e}")
            print("Please check your internet connection, model name, and GPU setup (if applicable).")
            self.model = None
            self.tokenizer = None

    def generate_response(self, query: str, contexts: List[Dict[str, str]]) -> str:
        """
        Generates a response based on the user query and retrieved contexts.

        Args:
            query (str): The original user query.
            contexts (List[Dict[str, str]]): A list of retrieved relevant text chunks,
                                              each dictionary containing 'text' and 'metadata'.

        Returns:
            str: The answer generated by the LLM.
        """
        if not self.model or not self.tokenizer:
            return "Error: LLM model not loaded. Cannot generate response."
        if not query or not query.strip():
            return "Please provide a valid query."

        # Construct the prompt for the LLM
        # Integrate the retrieved contexts into the prompt
        context_texts = [f"Source {i+1}: {c['text']}" for i, c in enumerate(contexts) if c and c.get('text')]
        context_str = "\n\n".join(context_texts) if context_texts else "No relevant context found."

        # Use Qwen's chat format
        # Detailed prompt engineering can significantly impact RAG performance
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context. If the answer is not in the context, say 'I don't have enough information to answer this question from the provided context.'"},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        try:
            # Encode the prompt
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            # Generate response
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=500, # Control the maximum length of generated text
                do_sample=True,    # Enable sampling
                temperature=0.7,   # Sampling temperature, controls randomness
                top_p=0.8,         # Top-p sampling, controls diversity
                eos_token_id=self.tokenizer.eos_token_id, # End-of-sequence token ID
                pad_token_id=self.tokenizer.pad_token_id # Padding token ID (prevents issues with long texts)
            )

            # Decode the generated IDs
            generated_ids = [
                output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()

        except Exception as e:
            print(f"Error during response generation: {e}")
            return f"Error generating response: {e}"

# Test area
if __name__ == "__main__":
    # 1. Initialize LLMGenerator
    print("--- Step 1: Initializing LLMGenerator ---")
    generator = LLMGenerator()

    if generator.model and generator.tokenizer:
        print("\n--- Step 2: Performing Response Generation Test ---")

        # Simulate retrieved contexts
        sample_contexts = [
            {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "fable.txt"}},
            {"text": "Artificial intelligence (AI) is intelligence demonstrated by machines.", "metadata": {"source": "wiki.pdf"}}
        ]

        # Test query
        test_query = "What is AI?"
        print(f"Query: {test_query}")
        print(f"Contexts: {len(sample_contexts)} chunks")

        # Generate response
        response = generator.generate_response(test_query, sample_contexts)
        print(f"\nGenerated Response:\n{response}")

        print("\n--- Testing with no relevant context ---")
        test_query_no_context = "What is the capital of France?"
        response_no_context = generator.generate_response(test_query_no_context, [])
        print(f"\nGenerated Response (no context):\n{response_no_context}")

    else:
        print("LLMGenerator initialization failed. Cannot perform generation test.")
