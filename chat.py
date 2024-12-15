from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
from typing import List
import numpy as np
import os
import pickle
import hashlib
from groq import Groq
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Groq API initialization
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize model and tokenizer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Directory paths and files
MD_DIRECTORY = r"C:\\Users\\MALIK COMPUTER\\Documents\\MyVault\\obsidian"
HASH_FILE = "file_hashes.pkl"
EMBEDDINGS_FILE = "embeddings.pkl"

# Utility functions
def clip_text_to_tokens(text, max_tokens):
    tokens = tokenizer.tokenize(text)
    clipped_tokens = tokens[:max_tokens]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(clipped_tokens))

def read_md_files(directory):
    texts = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f"File: {file_path}\n\n" + f.read()
                    texts.append(file_content)
    return texts

def compute_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def compute_directory_hash(directory):
    file_hashes = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                file_hashes[file_path] = compute_file_hash(file_path)
    return file_hashes

def save_hashes(filepath, hashes):
    with open(filepath, 'wb') as f:
        pickle.dump(hashes, f)

def load_hashes(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def has_directory_changed(directory, hash_filepath):
    new_hashes = compute_directory_hash(directory)
    if os.path.exists(hash_filepath):
        old_hashes = load_hashes(hash_filepath)
    else:
        old_hashes = {}

    if new_hashes != old_hashes:
        save_hashes(hash_filepath, new_hashes)
        return True
    return False

def save_embeddings(filepath, texts, embeddings):
    with open(filepath, 'wb') as f:
        pickle.dump((texts, embeddings), f)

def load_embeddings(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_relevant_texts(query, texts, embeddings, model, threshold=0.2):
    query_embedding = model.encode([query])
    similarities = model.similarity(query_embedding, embeddings)[0]
    relevant_indices = np.where(similarities >= threshold)[0]
    relevant_texts = [(texts[i], similarities[i]) for i in relevant_indices]
    relevant_texts.sort(key=lambda x: x[1], reverse=True)
    return relevant_texts

def stitch_relevant_texts(relevant_texts, max_size):
    stitched_text = "\n\n".join([text for text, _ in relevant_texts])
    return clip_text_to_tokens(stitched_text, max_size).strip()

def generate_response_from_groq(user_query, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"User query: {user_query}\n\nBased on the following information, provide relevant insights:\n\n{context}",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

# FastAPI models
class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 6000

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global md_texts, embeddings
    print("Application is starting up...")
    if has_directory_changed(MD_DIRECTORY, HASH_FILE):
        print("Changes detected in the markdown directory. Re-embedding...")
        md_texts = read_md_files(MD_DIRECTORY)
        embeddings = model.encode(md_texts)
        save_embeddings(EMBEDDINGS_FILE, md_texts, embeddings)
    else:
        print("No changes detected. Loading existing embeddings...")
        md_texts, embeddings = load_embeddings(EMBEDDINGS_FILE)
    
    yield  # Application continues to run
    
    print("Application is shutting down...")  # Cleanup actions can be added here

# Attach lifespan to the app
app = FastAPI(lifespan=lifespan)


@app.post("/query")
async def query_endpoint(queryRequest: QueryRequest):
    try:
        # Uncomment these lines when ready to use actual embedding
        relevant_texts = get_relevant_texts(queryRequest.query, md_texts, embeddings, model)
        stitched_text = stitch_relevant_texts(relevant_texts, queryRequest.max_tokens)
        #stitched_text = "Example test"
        response = generate_response_from_groq(queryRequest.query, stitched_text)
        return {"query": queryRequest.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

@app.get("/status")
def status():
    return {"status": "API is running", "documents_indexed": len(md_texts)}

@app.post("/reindex")
def reindex():
    if has_directory_changed(MD_DIRECTORY, HASH_FILE):
        md_texts = read_md_files(MD_DIRECTORY)
        embeddings = model.encode(md_texts)
        save_embeddings(EMBEDDINGS_FILE, md_texts, embeddings)
        return {"message": "Reindexing completed."}
    return {"message": "No changes detected. Reindexing skipped."}



@app.get("/", response_class=HTMLResponse)
async def serve_home():
    try:
        # Replace "static/index.html" with the path to your HTML file
        with open("static/index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>404 - Home Page Not Found</h1>",
            status_code=404
        )

