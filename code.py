import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Initialize the LLM (GPT model via OpenAI API)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure the environment variable is set

# Initialize the SentenceTransformer model for text embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimensionality of the embeddings (for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(dimension)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return "".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        print(f"Error extracting text from PDF ({pdf_file_path}): {e}")
        return ""

# Function to chunk the extracted text into smaller parts
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to embed text using the SentenceTransformer model
def embed_text(text_chunks):
    try:
        return embedding_model.encode(text_chunks, convert_to_numpy=True)
    except Exception as e:
        print(f"Error during embedding: {e}")
        return []

# Function to add embeddings to FAISS index
def add_to_faiss_index(embeddings):
    try:
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        print(f"Successfully added {embeddings_np.shape[0]} embeddings to FAISS index.")
    except Exception as e:
        print(f"Error adding to FAISS index: {e}")

# Function to perform a similarity search in the FAISS index
def search_in_faiss(query, top_k=5):
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        return indices[0], distances[0]
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return [], []

# Function to generate response from relevant chunks
def generate_response(query, relevant_chunks):
    s_words = {word.lower() for chunk in relevant_chunks for word in chunk.split() if word.lower().startswith('s')}
    
    if not s_words:
        return "Sorry, I couldn't find any words that start with 's' in the document."

    s_words_list = sorted(s_words)
    return f"Here are some words from the document that start with 's':\n- " + "\n- ".join(s_words_list)

# Main function to process PDF files, store embeddings, and respond to queries
def process_pdfs_and_query(pdf_files, query):
    all_chunks = []

    # Step 1: Data Ingestion
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            all_chunks.extend(chunk_text(text))

    if not all_chunks:
        return "No valid text extracted from the provided PDF files."

    # Step 2: Embed chunks and add to FAISS index
    embeddings = embed_text(all_chunks)
    if embeddings.size == 0:
        return "Failed to generate embeddings for the document."

    add_to_faiss_index(embeddings)

    # Step 3: Handle User Query
    indices, distances = search_in_faiss(query)
    relevant_chunks = [all_chunks[i] for i in indices] if indices.size > 0 else []

    # Step 4: Generate Response using extracted data
    response = generate_response(query, relevant_chunks)
    return response

# Example Usage:
pdf_files = ["C:/path/to/your/new_document.pdf"]  # Replace with your file paths
query = "Give me all words with starting characters is 's'?"
response = process_pdfs_and_query(pdf_files, query)
print(response)
