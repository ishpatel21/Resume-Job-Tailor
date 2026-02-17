"""
AI integration for resume tailoring using OpenAI or Google Gemini.
"""
from config import AI_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY
from typing import Optional
import os


class AIHandler:
    """Base class for AI providers."""
    
    def __init__(self):
        """Initialize AI handler based on configuration."""
        if AI_PROVIDER == "openai":
            self.provider = OpenAIHandler()
        elif AI_PROVIDER == "gemini":
            self.provider = GeminiHandler()
        else:
            raise ValueError(f"Unknown AI provider: {AI_PROVIDER}")
    
    def tailor_resume(self, resume_text: str, job_description: str) -> str:
        """
        Tailor resume based on job description.
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description text
            
        Returns:
            Tailored resume text
        """
        return self.provider.tailor_resume(resume_text, job_description)


class OpenAIHandler:
    """OpenAI API handler."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def tailor_resume(self, resume_text: str, job_description: str) -> str:
        """
        Use OpenAI to tailor resume.
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description text
            
        Returns:
            Tailored resume text
        """
        prompt = self._create_prompt(resume_text, job_description)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _create_prompt(resume_text: str, job_description: str) -> str:
        """Create prompt for OpenAI."""
        return f"""Please tailor the following resume to match the job description better.
        
Focus on:
1. Highlighting relevant skills and experiences
2. Using keywords from the job description
3. Reordering bullet points by relevance
4. Maintaining professional tone

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Please provide a tailored version of the resume that better matches the job description."""


class GeminiHandler:
    """Google Gemini API handler."""
    
    def __init__(self):
        """Initialize Gemini client."""
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")
    
    def tailor_resume(self, resume_text: str, job_description: str) -> str:
        """
        Use Google Gemini to tailor resume.
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description text
            
        Returns:
            Tailored resume text
        """
        prompt = self._create_prompt(resume_text, job_description)
        
        response = self.model.generate_content(prompt)
        return response.text
    
    @staticmethod
    def _create_prompt(resume_text: str, job_description: str) -> str:
        """Create prompt for Gemini."""
        return f"""Please tailor the following resume to match the job description better.
        
Focus on:
1. Highlighting relevant skills and experiences
2. Using keywords from the job description
3. Reordering bullet points by relevance
4. Maintaining professional tone

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Please provide a tailored version of the resume that better matches the job description."""
