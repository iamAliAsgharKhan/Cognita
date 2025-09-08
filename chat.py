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
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rank_bm25 import BM25Okapi
from datetime import datetime
from dateutil.parser import parse
import time
from typing import List, Dict, Optional
# --- Application Setup ---
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# --- Application State and Configuration ---
APP_CONFIG = {
    "md_directory": r"C:\obsidian",  # Default directory
    "hash_file": "file_hashes.pkl",
    "bm25_indices_path": "bm25_indices"  # <-- FIX: Added the missing key here
}

# Create the directory for BM25 indices if it doesn't exist
os.makedirs(APP_CONFIG["bm25_indices_path"], exist_ok=True)

# --- File System Event Handler ---
class MarkdownFileHandler(FileSystemEventHandler):
    # ... (No changes in this class)
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.md'):
            print(f"File created: {event.src_path}. Re-indexing document.")
            self.update_single_document(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.md'):
            print(f"File modified: {event.src_path}. Re-indexing document.")
            self.update_single_document(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.md'):
            print(f"File deleted: {event.src_path}. Removing from index.")
            self.remove_single_document(event.src_path)

    def update_single_document(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_content = f"File: {os.path.basename(file_path)}\n\n{content}"
            metadata = {'source': file_path}
            doc_id = file_path

            for model_name, model in loaded_models.items():
                collection_name = get_collection_name_for_model(model_name)
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    embedding = model.encode([doc_content]).tolist()
                    collection.add(embeddings=embedding, documents=[doc_content], metadatas=[metadata], ids=[doc_id])
                    print(f"  - Upserted into collection: {collection_name}")
                except ValueError:
                    print(f"  - Skipping non-existent collection: {collection_name}")
                except Exception as e:
                    print(f"  - Error updating collection {collection_name}: {e}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    def remove_single_document(self, file_path: str):
        doc_id = file_path
        for collection in chroma_client.list_collections():
            if collection.name.startswith("obsidian_"):
                try:
                    collection.delete(ids=[doc_id])
                    print(f"  - Deleted '{os.path.basename(file_path)}' from collection: {collection.name}")
                except Exception as e:
                    print(f"  - Error deleting from collection {collection.name}: {e}")

# --- File Watcher Management ---
file_observer = None

def start_file_watcher(directory: str):
    global file_observer
    if file_observer:
        file_observer.stop()
        file_observer.join()
        print("Stopped previous file watcher.")

    print(f"Starting file watcher for directory: {directory}")
    event_handler = MarkdownFileHandler()
    file_observer = Observer()
    file_observer.schedule(event_handler, directory, recursive=True)
    
    watcher_thread = threading.Thread(target=file_observer.start, daemon=True)
    watcher_thread.start()
    print("File watcher is running in the background.")

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

def extract_date_from_doc(file_path: str, content: str) -> Optional[int]:
    """
    Extracts a date from the filename, YAML frontmatter, or content of a doc.
    Handles various date formats.
    Returns the date as an integer in YYYYMMDD format, or None.
    """
    # Regex to find potential date strings
    date_pattern = re.compile(
        r'\b(\d{1,2}[-/]\w{3,}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'
    )
    
    # 1. Check filename
    filename = os.path.basename(file_path)
    match = date_pattern.search(filename)
    date_string = match.group(0) if match else None

    # 2. Check YAML frontmatter (simple check)
    if not date_string and content.strip().startswith('---'):
        frontmatter_match = re.search(r'---\s*\n(.*?)\n---', content, re.DOTALL)
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            date_line_match = re.search(r'^(date|created):\s*(.*)', frontmatter, re.IGNORECASE | re.MULTILINE)
            if date_line_match:
                potential_date = date_line_match.group(2).strip()
                if date_pattern.search(potential_date):
                    date_string = potential_date

    # 3. Check first few lines of content
    if not date_string:
        match = date_pattern.search(content[:150])
        if match:
            date_string = match.group(0)

    if date_string:
        try:
            # Use dateutil.parser to flexibly parse the date string
            dt = parse(date_string)
            return int(dt.strftime('%Y%m%d'))
        except (ValueError, OverflowError):
            return None # Invalid date format

    return None


def parse_time_from_query(query: str) -> Optional[Dict[str, int]]:
    """
    Parses a user query to find a year and returns a date range.
    Returns a dictionary {'start_date': YYYYMMDD, 'end_date': YYYYMMDD} or None.
    """
    # Check for a four-digit year (e.g., "in 2023", "tasks for 2022")
    year_match = re.search(r'\b(20\d{2})\b', query)
    year = None
    
    if year_match:
        year = int(year_match.group(1))
    elif "last year" in query.lower():
        year = datetime.now().year - 1
    elif "this year" in query.lower() or "current" in query.lower():
        year = datetime.now().year
        
    if year:
        start_date = int(f"{year}0101")
        end_date = int(f"{year}1231")
        return {"start_date": start_date, "end_date": end_date}
        
    return None


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
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        date_as_int = extract_date_from_doc(file_path, content)
                        # Get the file's last modification time as a timestamp
                        mod_time = os.path.getmtime(file_path)
                        docs.append({
                            "path": file_path,
                            "content": f"File: {file}\n\n{content}",
                            "date": date_as_int,
                            "modified_timestamp": mod_time
                        })
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
    return docs

def compute_directory_hash(directory: str):
    file_hashes = {}
    path = Path(directory)
    if not path.is_dir(): return {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        hasher = hashlib.md5()
                        hasher.update(f.read())
                        file_hashes[file_path] = hasher.hexdigest()
                except Exception as e:
                    print(f"Could not hash file {file_path}: {e}")
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
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def reciprocal_rank_fusion(results_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    fused_scores = {}
    for results in results_lists:
        for rank, doc_id in enumerate(results):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))

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
    global file_observer
    print("Application is starting up...")
    get_model(DEFAULT_MODEL_NAME)
    current_dir, hash_file = APP_CONFIG["md_directory"], APP_CONFIG["hash_file"]
    if has_directory_changed(current_dir, hash_file):
        print("Changes detected. Re-indexing with default model...")
        reindex_documents(DEFAULT_MODEL_NAME)
    else:
        print("No changes detected.")
    start_file_watcher(APP_CONFIG["md_directory"])
    yield
    print("Application is shutting down...")
    if file_observer:
        file_observer.stop()
        file_observer.join()
        print("File watcher stopped.")

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
    bm25_index_path = os.path.join(APP_CONFIG["bm25_indices_path"], f"{collection_name}.pkl")

    if not docs:
        if collection.count() > 0:
            collection.delete(ids=collection.get()['ids'])
        print("No markdown documents found to index. Collection cleared.")
        if os.path.exists(bm25_index_path):
            os.remove(bm25_index_path)
            print(f"Removed BM25 index: {bm25_index_path}")
        return 0

    documents = [doc['content'] for doc in docs]
    metadatas = []
    for doc in docs:
            meta = {'source': doc['path']}
            if doc['date']:
                meta['creation_date'] = doc['date']
            # Add the modified timestamp to the metadata
            if 'modified_timestamp' in doc:
                meta['modified_timestamp'] = doc['modified_timestamp']
            metadatas.append(meta)
    ids = [doc['path'] for doc in docs]
    if collection.count() > 0:
        collection.delete(ids=collection.get()['ids'])
    
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    print(f"Successfully indexed {len(documents)} documents into semantic collection '{collection_name}'.")

    print("Building BM25 keyword index...")
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(bm25_index_path, 'wb') as f:
        pickle.dump({'bm25': bm25, 'ids': ids}, f)
    print(f"Successfully built and saved BM25 index to '{bm25_index_path}'.")

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

    bm25_path = APP_CONFIG["bm25_indices_path"]
    if os.path.exists(bm25_path):
        for f in os.listdir(bm25_path):
            if f.endswith('.pkl'):
                os.remove(os.path.join(bm25_path, f))
        print("  - Deleted all BM25 indices.")

    start_file_watcher(APP_CONFIG["md_directory"])
            
    return {"message": "Directory updated successfully. All indexes have been cleared. Please re-index your documents."}

@app.get("/models")
async def get_available_models():
    return {"models": list(AVAILABLE_MODELS.keys())}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # --- 1. Setup ---
        model = get_model(request.model_name)
        collection_name = get_collection_name_for_model(request.model_name)
        
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except ValueError:
             raise HTTPException(status_code=404, detail=f"Collection for model '{request.model_name}' not found. Please re-index.")

        if collection.count() == 0:
            raise HTTPException(status_code=404, detail=f"No documents indexed for model '{request.model_name}'. Please re-index.")

        # --- 2. Time-Aware Query Analysis ---
        where_filter = None
        time_range = parse_time_from_query(request.query)
        if time_range:
            print(f"Time-based query detected. Filtering for range: {time_range}")
            where_filter = {
                "$and": [
                    {"creation_date": {"$gte": time_range["start_date"]}},
                    {"creation_date": {"$lte": time_range["end_date"]}}
                ]
            }

        # --- 3. Semantic Search (with optional time filter) ---
        print("Performing semantic search...")
        query_embedding = model.encode([request.query]).tolist()
        semantic_results = collection.query(
            query_embeddings=query_embedding, 
            n_results=10,  # Retrieve more to improve fusion
            where=where_filter,
            include=['metadatas']
        )
        semantic_doc_ids = semantic_results.get('ids', [[]])[0]

        # --- 4. Keyword Search ---
        print("Performing keyword search...")
        bm25_index_path = os.path.join(APP_CONFIG["bm25_indices_path"], f"{collection_name}.pkl")
        keyword_doc_ids = []
        try:
            with open(bm25_index_path, 'rb') as f:
                bm25_data = pickle.load(f)
                bm25 = bm25_data['bm25']
                doc_ids_map = bm25_data['ids']

            tokenized_query = request.query.lower().split()
            doc_scores = bm25.get_scores(tokenized_query)
            
            sorted_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:10]
            keyword_doc_ids = [doc_ids_map[i] for i in sorted_doc_indices]

        except FileNotFoundError:
            print(f"BM25 index not found for {collection_name}. Skipping keyword search.")
        
        # --- 5. Hybrid Fusion (RRF) ---
        print(f"Semantic results: {len(semantic_doc_ids)} docs. Keyword results: {len(keyword_doc_ids)} docs.")
        fused_results = reciprocal_rank_fusion([semantic_doc_ids, keyword_doc_ids])
        
        # --- 6. Recency Boost ---
        if fused_results:
            # Get metadata for all fused documents in one call for efficiency
            fused_ids = list(fused_results.keys())
            fused_metadatas_result = collection.get(ids=fused_ids, include=['metadatas'])
            
            # Create a map of doc_id -> metadata for easy lookup
            id_to_metadata = {
                fused_ids[i]: fused_metadatas_result['metadatas'][i] 
                for i in range(len(fused_ids))
            }
            
            current_time = time.time()
            boosted_scores = {}

            for doc_id, rrf_score in fused_results.items():
                metadata = id_to_metadata.get(doc_id)
                # Default score is the original RRF score
                final_score = rrf_score

                if metadata and 'modified_timestamp' in metadata:
                    modified_time = float(metadata['modified_timestamp'])
                    age_days = (current_time - modified_time) / (24 * 3600)
                    
                    # Apply a decay function: newer files get a bigger boost.
                    # The `0.1` factor controls how quickly the boost diminishes.
                    # A smaller value (e.g., 0.05) means a stronger, longer-lasting boost for recent files.
                    recency_boost = 1 / (1 + age_days * 0.1)
                    
                    # Add the boost to the original RRF score
                    final_score = rrf_score * (1 + recency_boost)
                
                boosted_scores[doc_id] = final_score
            
            # Sort documents by their new boosted scores to get the final ranking
            final_doc_ids = sorted(boosted_scores, key=boosted_scores.get, reverse=True)[:5]
            print(f"Fused and recency-boosted results: {len(final_doc_ids)} docs.")

        else:
            final_doc_ids = []
            print("No results after fusion.")


        # --- 7. Document Retrieval and Response Generation ---
        if not final_doc_ids:
            response_text = "No relevant context found in your documents."
            if time_range:
                year = time_range['start_date'] // 10000
                response_text = f"No relevant documents found for the year {year}."
            return QueryResponse(query=request.query, response=response_text, sources=[])

        retrieved_docs = collection.get(ids=final_doc_ids, include=['documents', 'metadatas'])
        
        relevant_docs = retrieved_docs.get('documents', [])
        relevant_metadatas = retrieved_docs.get('metadatas', [])
        source_paths = [meta['source'] for meta in relevant_metadatas if 'source' in meta]
        unique_filenames = sorted(list(set([os.path.basename(p) for p in source_paths])))
            
        stitched_text = stitch_relevant_texts(relevant_docs, request.max_tokens)
        response = generate_response_from_groq(request.query, stitched_text)
        
        return QueryResponse(query=request.query, response=response, sources=unique_filenames)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {e}") # Log the full error for debugging
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

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