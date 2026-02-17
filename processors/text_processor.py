"""
Text processing utilities for cleaning and analyzing resume and job description text.
"""
import re
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextProcessor:
    """Utilities for text processing and analysis."""
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords from text, removing stopwords.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords sorted by frequency
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words('english'))
        keywords = [
            token for token in tokens
            if token.isalpha() and token not in stop_words and len(token) > 2
        ]
        
        # Count frequency
        from collections import Counter
        freq = Counter(keywords)
        
        return [word for word, _ in freq.most_common(top_n)]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def find_matching_keywords(resume_text: str, job_description: str) -> dict:
        """
        Find keywords that appear in both resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with matching and missing keywords
        """
        resume_keywords = set(TextProcessor.extract_keywords(resume_text, top_n=50))
        job_keywords = set(TextProcessor.extract_keywords(job_description, top_n=50))
        
        matching = resume_keywords.intersection(job_keywords)
        missing = job_keywords - resume_keywords
        
        return {
            "matching": sorted(list(matching)),
            "missing": sorted(list(missing)),
            "total_job_keywords": len(job_keywords)
        }
