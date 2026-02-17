"""
Configuration settings for the Resume Job Tailor application.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # "openai" or "gemini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PDF Processing
PDF_MAX_PAGES = 2  # Most resumes are 1-2 pages
