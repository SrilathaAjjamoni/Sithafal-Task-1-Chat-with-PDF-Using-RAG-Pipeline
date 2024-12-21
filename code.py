import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load pre-trained embedding model
embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = [page.extract_text() for page in pdf]
    return text

# Step 2: Chunk text for embedding
def chunk_text(text, chunk_size=300):
    chunks = []
    for page_text in text:
        sentences = page_text.split('\n')
        for i in range(0, len(sentences), chunk_size):
            chunks.append(" ".join(sentences[i:i + chunk_size]))
    return chunks

# Step 3: Generate embeddings and store in FAISS
def store_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(np.array(embeddings))
    return index, embeddings, chunks

# Step 4: Query handling
def retrieve_relevant_chunks(query, index, embeddings, chunks, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Example Usage
pdf_path = "example.pdf"
query = "Unemployment rates based on degree type"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
index, embeddings, stored_chunks = store_embeddings(chunks)
relevant_chunks = retrieve_relevant_chunks(query, index, embeddings, stored_chunks)

# Print results
print("Relevant Chunks:\n", relevant_chunks)
