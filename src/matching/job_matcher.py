"""
Job description matching module for comparing resume text against job descriptions.
Implements two-layer scoring: keyword overlap (simple) and TF-IDF cosine similarity (advanced).
"""
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class JobMatcher:
    """
    Matches resume text against job descriptions using two scoring methods:
    1. Keyword Overlap: Fast, simple baseline (ATS-like)
    2. TF-IDF Cosine Similarity: More sophisticated matching
    """
    
    # Common skill keywords and technical terms
    SKILL_KEYWORDS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
        'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'git', 'ci/cd', 'jenkins', 'gitlab',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'fastapi',
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras',
        'data science', 'analytics', 'tableau', 'power bi', 'excel', 'statistics',
        'agile', 'scrum', 'kanban', 'jira', 'trello', 'asana',
        'api', 'rest', 'graphql', 'grpc', 'microservices', 'monolithic',
        'testing', 'pytest', 'junit', 'unittest', 'rspec', 'jest',
        'linux', 'unix', 'windows', 'macos', 'bash', 'shell', 'powershell',
        'communication', 'leadership', 'teamwork', 'problem-solving', 'project management',
        'html', 'css', 'xml', 'json', 'yaml', 'toml', 'markdown',
        'design patterns', 'oop', 'functional programming', 'clean code',
        'tcp', 'ip', 'http', 'https', 'websocket', 'network',
        'saas', 'paas', 'iaas', 'cloud computing', 'serverless',
        'agile', 'devops', 'sre', 'mlops', 'datalake', 'data warehouse'
    }
    
    # Stop words for filtering (extends NLTK's default)
    EXTENDED_STOPWORDS = set(stopwords.words('english')).union({
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'get', 'set', 'make', 'take', 'use', 'do', 'go', 'come', 'see',
        'job', 'role', 'position', 'candidate', 'team', 'company', 'work',
        'experience', 'responsibility', 'requirement', 'skill', 'able', 'required'
    })
    
    def __init__(self):
        """Initialize the job matcher."""
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for TF-IDF scoring. "
                "Install with: pip install scikit-learn"
            )
        # Vectorizer is created fresh for each comparison to avoid cache issues

    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for processing.
        
        Args:
            text: Raw text
            
        Returns:
            Normalized text (lowercased, whitespace cleaned)
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def _extract_job_keywords(job_description: str, top_n: int = 30) -> List[Tuple[str, int]]:
        """
        Extract important keywords from job description.
        Focuses on skills, requirements, and qualifications.
        
        Args:
            job_description: Job description text
            top_n: Number of top keywords to extract
            
        Returns:
            List of (keyword, frequency) tuples sorted by frequency
        """
        # Normalize text
        text = JobMatcher._normalize_text(job_description)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter: remove stopwords, short words, and non-alphabetic tokens
        filtered = [
            token for token in tokens
            if token.isalpha() 
            and token not in JobMatcher.EXTENDED_STOPWORDS
            and len(token) > 2
        ]
        
        # Count frequencies
        freq = Counter(filtered)
        
        # Get top keywords
        top_keywords = freq.most_common(top_n)
        
        return top_keywords
    
    @staticmethod
    def _find_matching_keywords(
        resume_text: str,
        job_keywords: List[Tuple[str, int]]
    ) -> Tuple[List[str], List[str]]:
        """
        Find which job keywords appear in the resume.
        
        Args:
            resume_text: Resume text
            job_keywords: List of job keywords to search for
            
        Returns:
            Tuple of (matching_keywords, missing_keywords)
        """
        # Normalize and tokenize resume
        normalized_resume = JobMatcher._normalize_text(resume_text)
        resume_tokens = set(word_tokenize(normalized_resume))
        
        # Extract keywords (just the words, not frequencies)
        job_keyword_words = [word for word, _ in job_keywords]
        
        matching = []
        missing = []
        
        for keyword in job_keyword_words:
            if keyword in resume_tokens:
                matching.append(keyword)
            else:
                missing.append(keyword)
        
        return matching, missing
    
    def calculate_keyword_overlap_score(
        self,
        resume_text: str,
        job_description: str
    ) -> Dict:
        """
        Calculate keyword overlap score (ATS-like baseline scoring).
        
        Method:
        1. Extract keywords from job description
        2. Count how many appear in resume
        3. Calculate percentage match
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with:
                - score: Percentage of job keywords found in resume (0-100)
                - matched_keywords: Keywords found in resume
                - missing_keywords: Keywords not found in resume
                - total_keywords: Total job keywords analyzed
        """
        if not isinstance(resume_text, str) or not isinstance(job_description, str):
            raise ValueError("Both inputs must be strings")
        
        if not resume_text.strip() or not job_description.strip():
            raise ValueError("Texts cannot be empty")
        
        # Extract job keywords
        job_keywords = self._extract_job_keywords(job_description, top_n=30)
        
        # Find matches
        matched, missing = self._find_matching_keywords(resume_text, job_keywords)
        
        # Calculate score as percentage
        total = len(job_keywords)
        matched_count = len(matched)
        score = (matched_count / total * 100) if total > 0 else 0
        
        return {
            'score': round(score, 2),
            'matched_keywords': matched,
            'missing_keywords': missing,
            'total_keywords': total,
            'method': 'keyword_overlap'
        }
    
    def calculate_tfidf_similarity(
        self,
        resume_text: str,
        job_description: str
    ) -> Dict:
        """
        Calculate TF-IDF cosine similarity score (advanced matching).
        
        Method:
        1. Create TF-IDF vectors for resume and job description
        2. Calculate cosine similarity between vectors
        3. Scale to 0-100 percentage
        
        This method captures semantic relevance and term importance better
        than simple keyword matching.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with:
                - score: Cosine similarity score (0-100)
                - interpretation: Human-readable interpretation
                - method: 'tfidf_cosine_similarity'
        """
        if not isinstance(resume_text, str) or not isinstance(job_description, str):
            raise ValueError("Both inputs must be strings")
        
        if not resume_text.strip() or not job_description.strip():
            raise ValueError("Texts cannot be empty")
        
        try:
            # Create a fresh vectorizer for this comparison
            # Note: Using unigrams (1,1) instead of bigrams for better matching
            # Bigrams create too sparse a matrix with often no overlap
            vectorizer = TfidfVectorizer(
                stop_words='english',
                lowercase=True,
                max_features=500,
                ngram_range=(1, 1),  # Unigrams only for better coverage
                min_df=1,
                max_df=1.0  # Don't filter out common words (they're the matches we want!)
            )
            
            # Fit on both texts and transform
            vectors = vectorizer.fit_transform([resume_text, job_description])
            
            # Calculate cosine similarity between the two documents
            similarity_matrix = cosine_similarity(vectors)
            similarity = similarity_matrix[0, 1]  # Similarity between resume and job
            
            # Scale to 0-100
            score = similarity * 100
            
            # Determine interpretation
            if score >= 80:
                interpretation = "Excellent match"
            elif score >= 60:
                interpretation = "Good match"
            elif score >= 40:
                interpretation = "Moderate match"
            elif score >= 20:
                interpretation = "Weak match"
            else:
                interpretation = "Poor match"
            
            return {
                'score': round(score, 2),
                'similarity_raw': round(similarity, 4),
                'interpretation': interpretation,
                'method': 'tfidf_cosine_similarity'
            }
        
        except Exception as e:
            raise RuntimeError(f"Error calculating TF-IDF similarity: {e}")
    
    def match_resume_to_job(
        self,
        resume_text: str,
        job_description: str,
        include_interpretation: bool = True
    ) -> Dict:
        """
        Perform comprehensive matching of resume against job description.
        Combines both keyword overlap and TF-IDF similarity scoring.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            include_interpretation: Whether to include human-readable interpretation
            
        Returns:
            Dictionary containing:
                - keyword_overlap: Keyword overlap score results
                - tfidf_similarity: TF-IDF cosine similarity results
                - overall_score: Average of both scores
                - matching_keywords: Keywords found in resume
                - missing_keywords: Keywords not found (top 10)
                - recommendations: List of suggestions for improvement
        """
        # Validate inputs
        if not isinstance(resume_text, str) or not isinstance(job_description, str):
            raise ValueError("Both inputs must be strings")
        
        if not resume_text.strip() or not job_description.strip():
            raise ValueError("Texts cannot be empty")
        
        # Calculate both scores
        keyword_score = self.calculate_keyword_overlap_score(resume_text, job_description)
        tfidf_score = self.calculate_tfidf_similarity(resume_text, job_description)
        
        # Calculate overall score (weighted average: 40% keyword, 60% TF-IDF)
        overall = (keyword_score['score'] * 0.4 + tfidf_score['score'] * 0.6)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            keyword_score,
            tfidf_score,
            overall
        )
        
        return {
            'keyword_overlap': keyword_score,
            'tfidf_similarity': tfidf_score,
            'overall_score': round(overall, 2),
            'matching_keywords': keyword_score['matched_keywords'],
            'missing_keywords': keyword_score['missing_keywords'][:10],  # Top 10 missing
            'recommendations': recommendations,
            'summary': self._generate_summary(overall, tfidf_score['interpretation'])
        }
    
    @staticmethod
    def _generate_recommendations(
        keyword_score: Dict,
        tfidf_score: Dict,
        overall_score: float
    ) -> List[str]:
        """
        Generate recommendations based on matching scores.
        
        Args:
            keyword_score: Keyword overlap results
            tfidf_score: TF-IDF similarity results
            overall_score: Overall combined score
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Keyword-based recommendations
        missing_count = len(keyword_score['missing_keywords'])
        if missing_count > 0:
            top_missing = keyword_score['missing_keywords'][:5]
            recommendations.append(
                f"Add these keywords to your resume: {', '.join(top_missing)}"
            )
        
        # Overall score recommendations
        if overall_score < 50:
            recommendations.append(
                "Resume has limited overlap with this job description. "
                "Consider adding relevant experience or skills."
            )
        elif overall_score < 70:
            recommendations.append(
                "Good match overall, but some keywords are missing. "
                "Review job description and highlight similar experience."
            )
        else:
            recommendations.append(
                "Strong match! Your resume aligns well with the job requirements."
            )
        
        # TF-IDF based recommendations
        if tfidf_score['score'] < 40:
            recommendations.append(
                "Consider restructuring your resume to emphasize relevant experience "
                "and skills that match this specific role."
            )
        
        return recommendations
    
    @staticmethod
    def _generate_summary(overall_score: float, interpretation: str) -> str:
        """
        Generate a human-readable summary of the match.
        
        Args:
            overall_score: Overall combined score
            interpretation: TF-IDF interpretation
            
        Returns:
            Summary string
        """
        if overall_score >= 80:
            return "🟢 Excellent match - Your resume is well-aligned with the job"
        elif overall_score >= 60:
            return "🟡 Good match - Your resume covers most requirements"
        elif overall_score >= 40:
            return "🟠 Moderate match - Some gaps between resume and job requirements"
        else:
            return "🔴 Limited match - Significant differences between resume and job"


class QuickMatcher:
    """
    Lightweight matcher for quick keyword matching without sklearn dependency.
    Fallback option if sklearn is not available.
    """
    
    @staticmethod
    def quick_keyword_match(resume_text: str, job_description: str) -> Dict:
        """
        Perform quick keyword matching without sklearn.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with keyword matching results
        """
        # Simple tokenization and comparison
        resume_tokens = set(word.lower() for word in re.findall(r'\b\w+\b', resume_text))
        job_tokens = set(word.lower() for word in re.findall(r'\b\w+\b', job_description))
        
        # Remove common stopwords
        stop_words = set(stopwords.words('english'))
        resume_tokens = {t for t in resume_tokens if t not in stop_words and len(t) > 2}
        job_tokens = {t for t in job_tokens if t not in stop_words and len(t) > 2}
        
        # Calculate overlap
        matching = resume_tokens.intersection(job_tokens)
        missing = job_tokens - resume_tokens
        
        score = (len(matching) / len(job_tokens) * 100) if job_tokens else 0
        
        return {
            'score': round(score, 2),
            'matched_keywords': sorted(list(matching)),
            'missing_keywords': sorted(list(missing))[:10],
            'total_keywords': len(job_tokens),
            'method': 'quick_keyword_match'
        }
