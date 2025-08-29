import os
from dotenv import load_dotenv
import pickle
import hashlib
import re
from typing import List, Dict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
from groq import Groq

# --- Application Setup ---
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# --- Application State and Configuration ---
APP_CONFIG = {
    "md_directory": r"C:\obsidian",  # Default directory
    "hash_file": "file_hashes.pkl"
}

# --- Model Management ---
AVAILABLE_MODELS = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
    "MPNet": "sentence-transformers/all-mpnet-base-v2"
}
DEFAULT_MODEL_NAME = "MiniLM-L6"
loaded_models: Dict[str, SentenceTransformer] = {}

def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not available.")
    if model_name not in loaded_models:
        print(f"Loading model '{model_name}'...")
        model_path = AVAILABLE_MODELS[model_name]
        loaded_models[model_name] = SentenceTransformer(model_path)
        print(f"Model '{model_name}' loaded successfully.")
    return loaded_models[model_name]

# --- Chroma DB Management ---
chroma_client = chromadb.PersistentClient(path="my_chroma_db")

def get_collection_name_for_model(model_name: str) -> str:
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name).lower()
    return f"obsidian_{sanitized_name}"

# --- Utility Functions ---
def clip_text_to_tokens(text, max_tokens):
    tokens = tokenizer.tokenize(text)
    clipped_tokens = tokens[:max_tokens]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(clipped_tokens))

def read_md_files(directory: str):
    docs = []
    path = Path(directory)
    if not path.is_dir(): return []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs.append({"path": file_path, "content": f"File: {file}\n\n{content}"})
    return docs

def compute_directory_hash(directory: str):
    file_hashes = {}
    path = Path(directory)
    if not path.is_dir(): return {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    hasher = hashlib.md5()
                    hasher.update(f.read())
                    file_hashes[file_path] = hasher.hexdigest()
    return file_hashes

def save_hashes(filepath: str, hashes: dict):
    with open(filepath, 'wb') as f: pickle.dump(hashes, f)

def load_hashes(filepath: str):
    if not os.path.exists(filepath): return {}
    with open(filepath, 'rb') as f: return pickle.load(f)

def has_directory_changed(directory: str, hash_filepath: str):
    new_hashes, old_hashes = compute_directory_hash(directory), load_hashes(hash_filepath)
    if new_hashes != old_hashes:
        save_hashes(hash_filepath, new_hashes)
        return True
    return False

def stitch_relevant_texts(relevant_docs, max_size):
    stitched_text = "\n\n".join(relevant_docs)
    return clip_text_to_tokens(stitched_text, max_size).strip()

def generate_response_from_groq(user_query, context):
    chat_completion = client.chat.completions.create(
        messages=[{ "role": "user", "content": f"User query: {user_query}\n\nBased on the following information, provide relevant insights:\n\n{context}"}],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    model_name: str = DEFAULT_MODEL_NAME
    max_tokens: int = 6000

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[str]

class ReindexRequest(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME

class DirectoryRequest(BaseModel):
    path: str

# --- FastAPI App and Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    get_model(DEFAULT_MODEL_NAME)
    current_dir, hash_file = APP_CONFIG["md_directory"], APP_CONFIG["hash_file"]
    if has_directory_changed(current_dir, hash_file):
        print("Changes detected. Re-indexing with default model...")
        reindex_documents(DEFAULT_MODEL_NAME)
    else:
        print("No changes detected.")
    yield
    print("Application is shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Core Logic Functions ---
def reindex_documents(model_name: str):
    directory = APP_CONFIG["md_directory"]
    print(f"Starting re-indexing for model '{model_name}' on directory '{directory}'...")
    model = get_model(model_name)
    collection_name = get_collection_name_for_model(model_name)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    docs = read_md_files(directory)

    if not docs:
        if collection.count() > 0:
            collection.delete(ids=collection.get()['ids'])
        print("No markdown documents found to index. Collection cleared.")
        return 0

    documents = [doc['content'] for doc in docs]
    metadatas = [{'source': doc['path']} for doc in docs]
    ids = [doc['path'] for doc in docs]

    # --- Indexing Logic Explanation ---
    # The following 'delete' and 'add' sequence is intentional and correct.
    # While using only 'add' (as an upsert) is possible, it fails to
    # remove documents from the index that have been deleted from the source folder.
    # This two-step process ensures the database is a perfect mirror of the
    # files on disk, preventing stale or outdated information.
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    
    print(f"Successfully indexed {len(documents)} documents into collection '{collection_name}'.")
    return len(documents)

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>404 - Not Found</h1>", status_code=404)

@app.get("/directory")
async def get_current_directory():
    return {"path": APP_CONFIG["md_directory"]}

@app.post("/directory")
async def set_directory(request: DirectoryRequest):
    new_path = Path(request.path)
    if not new_path.is_dir():
        raise HTTPException(status_code=400, detail=f"The provided path is not a valid directory: {request.path}")
    
    APP_CONFIG["md_directory"] = request.path
    if os.path.exists(APP_CONFIG["hash_file"]):
        os.remove(APP_CONFIG["hash_file"])
    
    print(f"Markdown directory updated. Deleting all Chroma DB collections...")
    for collection in chroma_client.list_collections():
        if collection.name.startswith("obsidian_"):
            chroma_client.delete_collection(name=collection.name)
            print(f"  - Deleted collection: {collection.name}")
            
    return {"message": "Directory updated successfully. All indexes have been cleared. Please re-index your documents."}

@app.get("/models")
async def get_available_models():
    return {"models": list(AVAILABLE_MODELS.keys())}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        model = get_model(request.model_name)
        collection_name = get_collection_name_for_model(request.model_name)
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except ValueError:
             raise HTTPException(status_code=404, detail=f"Collection for model '{request.model_name}' not found. Please re-index.")

        if collection.count() == 0:
            raise HTTPException(status_code=404, detail=f"No documents indexed for model '{request.model_name}'. Please re-index.")

        query_embedding = model.encode([request.query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding, 
            n_results=5,
            include=['metadatas', 'documents']
        )
        
        relevant_docs = results.get('documents', [[]])[0]

        if not relevant_docs:
            return QueryResponse(query=request.query, response="No relevant context found in your documents.", sources=[])
        
        relevant_metadatas = results.get('metadatas', [[]])[0]
        source_paths = [meta['source'] for meta in relevant_metadatas if 'source' in meta]
        unique_filenames = sorted(list(set([os.path.basename(p) for p in source_paths])))
            
        stitched_text = stitch_relevant_texts(relevant_docs, request.max_tokens)
        response = generate_response_from_groq(request.query, stitched_text)
        
        return QueryResponse(query=request.query, response=response, sources=unique_filenames)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

@app.get("/status")
def status():
    collections_data = {c.name: c.count() for c in chroma_client.list_collections()}
    return {"status": "API is running", "indexed_collections": collections_data}

@app.post("/reindex")
def reindex(request: ReindexRequest):
    try:
        count = reindex_documents(request.model_name)
        current_dir, hash_file = APP_CONFIG["md_directory"], APP_CONFIG["hash_file"]
        save_hashes(hash_file, compute_directory_hash(current_dir))
        return {"message": f"Re-indexing completed for model '{request.model_name}'. {count} documents indexed."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during re-indexing: {e}")