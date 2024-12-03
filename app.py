import os
from typing import List
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import time
import hashlib
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration from environment variables with defaults
PDFS_DIRECTORY = os.getenv("PDFS_DIRECTORY", "./knowledge_base")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:latest")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
ALLOWED_EXTENSIONS = {'.pdf'}

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    context: List[str]

class RAGSystem:
    def __init__(self):
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure directories exist
        os.makedirs(PDFS_DIRECTORY, exist_ok=True)
        
        # Initial load of PDFs
        self.load_pdfs_from_directory()
    
    def generate_chunk_id(self, filepath: str, chunk_content: str, chunk_index: int) -> str:
        """Generate a unique ID for a chunk based on its content and metadata"""
        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
        timestamp = int(time.time())
        return f"{os.path.basename(filepath)}_{chunk_index}_{content_hash}_{timestamp}"

    def load_pdfs_from_directory(self):
        """Load PDFs from the knowledge base directory"""
        for filename in os.listdir(PDFS_DIRECTORY):
            if os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS:
                try:
                    self.process_pdf(os.path.join(PDFS_DIRECTORY, filename))
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    
    def process_pdf(self, filepath: str, chunk_size: int = 1000):
        """Extract text from PDF and add to Chroma with chunking"""
        try:
            reader = PdfReader(filepath)
            full_text = ""
            
            # Extract text from all pages
            for page in reader.pages:
                full_text += page.extract_text() + " "
            
            # Split text into chunks
            words = full_text.split()
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
            
            # Generate embeddings in batches
            embeddings = self.embedder.encode(chunks).tolist()
            
            # Generate unique IDs for chunks
            chunk_ids = [
                self.generate_chunk_id(filepath, chunk, i)
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to Chroma with unique IDs
            if chunks:  # Only proceed if there are chunks to add
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=chunk_ids
                )
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context from Chroma"""
        try:
            query_embedding = self.embedder.encode([query])[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results['documents'][0]
        except Exception as e:
            raise Exception(f"Error retrieving context: {str(e)}")
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Ollama API"""
        prompt = f"""Context: {' '.join(context)}

Question: {query}
Answer in a clear, concise manner based on the provided context. If the context does not contain sufficient information, acknowledge that."""
        
        try:
            response = requests.post(
                OLLAMA_URL, 
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 300
                    }
                },
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            return response.json()['response'].strip()
        except requests.RequestException as e:
            raise Exception(f"Error generating response: {str(e)}")

# FastAPI App
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API using FastAPI, ChromaDB, and Ollama",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System
rag_system = RAGSystem()

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end of file
    size = file.file.tell()
    file.file.seek(0)  # Reset file pointer
    
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE/1024/1024}MB"
        )

@app.get("/")
async def read_root():
    """Serve the index.html file"""
    return FileResponse("index.html")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file uploads"""
    try:
        # Validate file
        validate_file(file)
        
        # Save uploaded file
        filepath = os.path.join(PDFS_DIRECTORY, file.filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the uploaded PDF
        rag_system.process_pdf(filepath)
        
        return JSONResponse(
            content={"filename": file.filename, "status": "uploaded and processed"},
            status_code=200
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def handle_query(query_request: QueryRequest):
    """Process user query and generate response"""
    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Retrieve relevant context
        context = rag_system.retrieve_context(query_request.query)
        
        # Generate response
        response = rag_system.generate_response(query_request.query, context)
        
        return QueryResponse(
            response=response,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-form")
async def handle_query_form(query: str = Form(...)):
    """Process user query from form data and generate response"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        # Retrieve relevant context
        context = rag_system.retrieve_context(query)
        
        # Generate response
        response = rag_system.generate_response(query, context)
        
        return JSONResponse(
            content={
                "response": response,
                "context": context
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
