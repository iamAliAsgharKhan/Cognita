
# Cognita

<p align="center">
  <img src="./images/logo.png" alt="Cognita Logo" width="400">
</p>
<p align="center">
  A sophisticated, conversational AI web application that allows you to chat with your local Obsidian notes. It uses a state-of-the-art hybrid search system (semantic + keyword) with Reciprocal Rank Fusion (RRF) and a recency boost to provide incredibly accurate, contextually-aware answers, backed by a persistent vector database and powered by the high-speed Groq LLM API.
</p>

<p align="center">
  <a href="https://buymeacoffee.com/ialiasghardev" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>

The application features a modern chat interface, dynamic AI model configuration, and provides source citations for its answers. It now watches your notes directory in real-time, automatically keeping its knowledge base perfectly in sync with your work.

![New Application Screenshot](./images/screenshot1.png)

---

## ✨ Key Features

-   **Automatic Real-time Indexing**: Cognita watches your notes directory for changes. Create, modify, or delete a note, and the knowledge base is updated instantly and automatically in the background.
-   **Advanced Hybrid Search**: Goes beyond simple vector search by combining traditional keyword-based search (BM25) with modern semantic search. Results are fused using **Reciprocal Rank Fusion (RRF)** for state-of-the-art retrieval accuracy.
-   **Recency Boost**: Prioritizes your most recent notes. The ranking algorithm gives a dynamic boost to documents that have been recently modified, ensuring you get the most current information.
-   **Time-Aware Queries**: Ask questions related to specific timeframes. Queries like "What were my goals in 2023?" or "Summarize my meetings from last year" will automatically filter your notes by the relevant dates found within the documents.
-   **Persistent Vector Storage with Chroma DB**: Your document embeddings are saved, scalable, and load instantly on startup, powered by the robust Chroma DB.
-   **Dynamic Embedding Model Selection**: Choose from multiple `sentence-transformer` models in the UI. The backend dynamically loads them and stores embeddings in separate, model-specific collections.
-   **Configurable Source Directory**: Easily change the source directory of your Obsidian notes directly from the UI.
-   **Source-Aware Responses (Citations)**: Build trust in your AI's answers. The model cites the exact notes used to generate a response, allowing for easy verification.
-   **Conversational Chat Interface**: Interact with your knowledge base through an intuitive, chat-bubble UI.
-   **Clean Settings Modal**: Key settings like the source directory and model selection are managed in a clean, unobtrusive modal window.

---

## 🛠️ Tech Stack

-   **Backend**: FastAPI
-   **LLM**: Groq (Llama-3.3-70b-versatile)
-   **Vector Database**: Chroma DB
-   **Keyword Search**: Rank-BM25
-   **File Monitoring**: Watchdog
-   **Embedding Models**: Sentence Transformers
-   **Frontend**: HTML5, CSS3, Vanilla JavaScript
-   **API Key Management**: python-dotenv
-   **Date Parsing**: python-dateutil

---

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+**
2.  **Groq API Key**: Obtain a free API key from [Groq](https://console.groq.com/keys).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/iamAliAsgharKhan/Cognita.git
    cd Cognita
    ```

2.  **Install the required Python libraries:**
    *(It is recommended to use a virtual environment)*
    ```bash
    pip install "fastapi[all]" sentence-transformers groq chromadb python-dotenv transformers watchdog rank_bm25 python-dateutil
    ```

3.  **Set up your environment variables:**
    Create a file named `.env` in the project's root directory. Copy the contents of `example.env` into it and add your Groq API key:
    ```.env
    GROQ_API_KEY="gsk_YourSecretKeyHere"
    ```

### Running the Application

1.  **Start the FastAPI server:**
    ```bash
    uvicorn chat:app --reload
    ```
    On the first run, the app will download the default embedding model, create a `my_chroma_db` directory for the vector database, and begin indexing your notes.

2.  **Access the web interface:**
    Open your browser and navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## ⚙️ How It Works

1.  **Initial Indexing**: On first run with a new directory, Cognita scans all `.md` files. For the chosen embedding model, it generates:
    *   **Vector Embeddings**: Stored in a Chroma DB collection for semantic search.
    *   **Keyword Index**: A BM25 index is created and saved for efficient keyword search.
2.  **Live Monitoring**: A file watcher runs in the background, monitoring your notes directory. Any time you save, create, or delete a markdown file, it triggers an incremental update to both the Chroma DB collection and the BM25 index.
3.  **Querying**:
    -   A user submits a query via the chat interface.
    -   **Time Analysis**: The query is first checked for time-based phrases (e.g., "2023", "last year") to create a date filter.
    -   **Hybrid Search**: The application performs two searches in parallel:
        1.  **Semantic Search**: The query is embedded, and Chroma DB finds the most semantically similar notes (optionally filtered by date).
        2.  **Keyword Search**: The BM25 index is queried to find notes with the best keyword overlap.
    -   **Fusion & Ranking**: The results from both searches are combined using the **Reciprocal Rank Fusion (RRF)** algorithm to produce a unified, more accurate list.
    -   **Recency Boost**: This fused list is then re-ranked, giving a boost to notes that have been modified more recently.
    -   **Context Building**: The content of the top-ranked notes is retrieved and stitched together to form a rich context.
    -   **LLM Response**: This context and the original query are sent to the Groq LLM API.
    -   The LLM's response and the list of source filenames are sent back to the UI and displayed.

---

## ↔️ API Endpoints

-   `GET /`: Serves the main HTML interface.
-   `POST /query`: Submits a query.
    -   **Request**: `{"query": "string", "model_name": "string"}`
    -   **Response**: `{"query": "string", "response": "string", "sources": ["file1.md", "file2.md"]}`
-   `POST /reindex`: Triggers a full manual re-index of the source directory.
    -   **Request**: `{"model_name": "string"}`
-   `GET /models`: Returns a list of available embedding models.
-   `GET /directory`: Returns the current notes directory path.
-   `POST /directory`: Sets a new notes directory path and clears all existing indexes.
-   `GET /status`: Returns the status and document count of all indexed collections.

---

## 🔮 Future Improvements

-   **Implement Streaming Responses**: Stream tokens from Groq as they are generated for a much faster perceived response time.
-   **Improve Document Chunking**: Split large files into smaller, more focused chunks before embedding to improve retrieval accuracy.
-   **Maintain Conversation History**: Allow for follow-up questions by sending the recent chat history to the LLM as additional context.

---

## ❤️ Support This Project

If you find Cognita useful and want to support its development, please consider buying me a coffee!

<a href="https://buymeacoffee.com/ialiasghardev" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.