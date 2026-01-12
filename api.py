
# IMPORTS

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import os

# Import the RAG chain and PDF path from your main.py
from main import qa_chain, PDF_PATH



# CREATE FASTAPI APP

app = FastAPI(title="ML Textbook RAG API")



# CORS SETUP (allows frontend to connect)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this later)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)



# REQUEST/RESPONSE MODELS

# Model for incoming question request
class QuestionRequest(BaseModel):
    question: str

# Model for source document in response
class SourceDocument(BaseModel):
    page: int
    content: str

# Model for the final answer response
class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]



# API ENDPOINTS


# Root endpoint - just to check if API is running
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "ML Textbook RAG API is running!",
        "endpoints": {
            "GET /pdf": "Get the PDF file",
            "POST /ask": "Ask a question about the textbook",
            "GET /health": "Check API health"
        }
    }


# Endpoint to serve the PDF file
@app.get("/pdf")
async def get_pdf():
    """Serves the PDF file to the frontend"""
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Return the PDF file
    return FileResponse(
        PDF_PATH, 
        media_type="application/pdf", 
        filename="MLBook.pdf"
    )


# Main endpoint to ask questions
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Receives a question and returns an AI-generated answer with sources
    """
    # Check if RAG system is initialized
    if qa_chain is None:
        raise HTTPException(
            status_code=500, 
            detail="RAG system not initialized"
        )
    
    # Validate that question is not empty
    if not request.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    try:
        # Use the RAG chain to get answer
        result = qa_chain.invoke({"query": request.question})
        
        # Extract and format source documents
        sources = []
        for doc in result.get('source_documents', []):
            sources.append(SourceDocument(
                page=doc.metadata.get('page', 0),
                # Truncate content to 300 characters
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            ))
        
        # Return the answer and sources
        return AnswerResponse(
            answer=result['result'],
            sources=sources
        )
        
    except Exception as e:
        # Handle any errors that occur
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API and RAG system are working"""
    return {
        "status": "healthy",
        "rag_initialized": qa_chain is not None,
        "pdf_exists": os.path.exists(PDF_PATH)
    }



# RUN THE SERVER

if __name__ == "__main__":
    import uvicorn
    
    print(f" Server will run on: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/docs")
   
    
    # Run on localhost:8000 (reload=False to avoid the warning)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)