import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Configuration
PDFS_DIRECTORY = "./knowledge_base"
CHROMA_PATH = "./chroma_db"
MODEL_PATH = "./models/Llama-3.2-3B-Instruct-f16.gguf"  # Update with your Llama model path

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
        
        # Initialize Llama model
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,  # Context window
            n_batch=512  # Batch size
        )
        
        # Initial load of PDFs
        self.load_pdfs_from_directory()
    
    def load_pdfs_from_directory(self):
        """Load PDFs from the knowledge base directory"""
        if not os.path.exists(PDFS_DIRECTORY):
            os.makedirs(PDFS_DIRECTORY)
        
        for filename in os.listdir(PDFS_DIRECTORY):
            if filename.endswith('.pdf'):
                self.process_pdf(os.path.join(PDFS_DIRECTORY, filename))
    
    def process_pdf(self, filepath):
        """Extract text from PDF and add to Chroma"""
        reader = PdfReader(filepath)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts).tolist()
        
        # Add to Chroma with unique IDs
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[f"{filename}_{i}" for filename in [os.path.basename(filepath)]]
            )
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context from Chroma"""
        query_embedding = self.embedder.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]
    
    def generate_response(self, query, context):
        """Generate response using Llama with retrieved context"""
        # Combine context and query
        prompt = f"""Context: {' '.join(context)}
        
Question: {query}
Answer:"""
        
        # Generate response
        response = self.llm(
            prompt, 
            max_tokens=300,  # Adjust as needed
            stop=["Question:", "\n"],
            echo=False
        )
        
        return response['choices'][0]['text'].strip()

# FastAPI App
app = FastAPI()

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System
rag_system = RAGSystem()

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file uploads"""
    try:
        # Save uploaded file
        filepath = os.path.join(PDFS_DIRECTORY, file.filename)
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the uploaded PDF
        rag_system.process_pdf(filepath)
        
        return {"filename": file.filename, "status": "uploaded and processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def handle_query(query: str):
    """Process user query and generate response"""
    try:
        # Retrieve relevant context
        context = rag_system.retrieve_context(query)
        
        # Generate response
        response = rag_system.generate_response(query, context)
        
        return {
            "response": response,
            "context": context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)