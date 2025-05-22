from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
import logging
import time
import json
import traceback
from pathlib import Path
import sys

# Import the resume scanner model
from scp import TFIDFEnhancedResumeScanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="CVScan API",
    description="API for scanning and matching resumes with job titles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use existing directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
CV_DIR = BASE_DIR / "cv_dummy"

# Global scanner instance
scanner = None

# Background task to initialize the scanner
def initialize_scanner():
    global scanner
    try:
        logging.info("Initializing resume scanner...")
        scanner = TFIDFEnhancedResumeScanner(
            input_folder=str(INPUT_DIR),
            output_folder=str(OUTPUT_DIR),
            cv_folder=str(CV_DIR)
        )
        
        # Check if input files exist
        input_files_exist = all(
            os.path.exists(os.path.join(INPUT_DIR, f)) 
            for f in ['01_people.csv', '02_abilities.csv', '03_education.csv', '04_experience.csv', '05_person_skills.csv']
        )
        
        if not input_files_exist:
            error_msg = "Input CSV files are missing. Please add the required CSV files to the input directory."
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load data
        scanner.load_data()
        
        # Create TF-IDF vectors
        scanner.create_tfidf_vectors()
        
        # Create embeddings
        scanner.create_embeddings()
        
        logging.info("Resume scanner initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing scanner: {str(e)}")
        logging.error(traceback.format_exc())
        scanner = None
        raise

# Response models
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool

class MatchResult(BaseModel):
    title: str
    similarity_percentage: float
    embedding_text: Optional[str] = None

class ScanResponse(BaseModel):
    matches: List[MatchResult]
    avg_similarity: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and the model is loaded"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": scanner is not None
    }

@app.post("/initialize", response_model=dict)
async def initialize_model(background_tasks: BackgroundTasks):
    """Initialize the resume scanner model in the background"""
    if scanner is not None:
        return {"message": "Model already initialized"}
    
    background_tasks.add_task(initialize_scanner)
    return {"message": "Model initialization started in background"}

@app.post("/scan/text", response_model=ScanResponse)
async def scan_text(text: str = Form(...), top_n: int = Form(5)):
    """Scan resume text and match against job titles"""
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    try:
        # Match the text against job titles
        results = scanner.match_text(text, top_n=top_n)
        
        # Convert DataFrame to list of dictionaries
        matches = []
        for _, row in results['matches'].iterrows():
            match_dict = {
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage']
            }
            if 'embedding_text' in row:
                match_dict["embedding_text"] = row['embedding_text']
            matches.append(match_dict)
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning text: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan/file", response_model=ScanResponse)
async def scan_file(file: UploadFile = File(...), top_n: int = Form(5)):
    """Scan resume file and match against job titles"""
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or Word documents.")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    temp_file_path = temp_file.name
    
    try:
        # Save uploaded file to temp file
        with temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Process the resume file
        results = scanner.process_resume_file(temp_file_path, top_n=top_n)
        
        # Convert DataFrame to list of dictionaries
        matches = []
        for _, row in results['matches'].iterrows():
            match_dict = {
                "title": row['title'],
                "similarity_percentage": row['similarity_percentage']
            }
            if 'embedding_text' in row:
                match_dict["embedding_text"] = row['embedding_text']
            matches.append(match_dict)
        
        return {
            "matches": matches,
            "avg_similarity": results['avg_similarity']
        }
    except Exception as e:
        logging.error(f"Error scanning file: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/evaluate", response_model=dict)
async def evaluate_model(k_values: List[int] = [3, 5, 10], relevance_threshold: Optional[float] = None):
    """Evaluate the model using ground truth data"""
    if scanner is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Call /initialize first.")
    
    try:
        # Run evaluation
        metrics = scanner.evaluate_model(k_values=k_values, relevance_threshold=relevance_threshold)
        
        if metrics is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Evaluation failed. Check server logs for details."}
            )
        
        return {
            "message": "Evaluation completed successfully",
            "metrics": metrics
        }
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to initialize the scanner
@app.on_event("startup")
async def startup_event():
    # Check if input directories exist
    for directory in [INPUT_DIR, OUTPUT_DIR, CV_DIR]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")
    
    # Check if input files exist
    input_files_exist = all(
        os.path.exists(os.path.join(INPUT_DIR, f)) 
        for f in ['01_people.csv', '02_abilities.csv', '03_education.csv', '04_experience.csv', '05_person_skills.csv']
    )
    
    if not input_files_exist:
        logging.warning("Input CSV files are missing. The API will start, but initialization will fail until files are provided.")
        logging.warning(f"Please add the required CSV files to: {INPUT_DIR}")
    else:
        # Initialize the scanner in the background
        try:
            initialize_scanner()
        except Exception as e:
            logging.error(f"Failed to initialize scanner on startup: {str(e)}")
            logging.error("The API will start, but you'll need to call /initialize endpoint manually.")

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    global scanner
    if scanner is not None:
        scanner.cleanup()
        scanner = None
        logging.info("Scanner resources cleaned up")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)