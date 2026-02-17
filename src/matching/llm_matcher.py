"""
LLM-based job matching analysis using OpenAI or Google Gemini.
Provides intelligent skill assessment and detailed matching insights.
"""
import json
import os
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import re

# Try importing both LLM providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class LLMMatchResult:
    """Result from LLM-based matching."""
    match_percentage: int
    missing_keywords: List[str]
    weak_areas: List[str]
    strengths: List[str]
    detailed_analysis: str
    recommendation: str
    top_skills_required: List[str]
    top_skills_matched: List[str]


class LLMJobMatcher:
    """
    LLM-based job description matcher using OpenAI or Google Gemini.
    Provides intelligent skill assessment beyond simple keyword matching.
    """
    
    SYSTEM_PROMPT = """You are an expert recruiter and technical hiring manager with 15+ years of experience.
Your task is to analyze job descriptions and resumes to identify skill matches, gaps, and provide insights.
Be concise, practical, and focus on actionable feedback."""

    ANALYSIS_PROMPT_TEMPLATE = """Analyze this job description and resume excerpt for skill matching.

JOB DESCRIPTION:
{jd}

RESUME EXCERPT:
{resume}

Please provide a detailed analysis in JSON format with:
1. match_percentage: Overall skill match (0-100)
2. missing_keywords: List of 3-5 required skills NOT found in resume
3. weak_areas: List of 2-3 areas where candidate is weak compared to requirements
4. strengths: List of 2-3 areas where candidate is strong
5. top_skills_required: List of 5-8 most important skills from the job description
6. top_skills_matched: List of skills from requirement that ARE in the resume
7. detailed_analysis: 2-3 sentence paragraph with specific observations
8. recommendation: 1-2 sentence hiring recommendation

Return ONLY valid JSON, no markdown, no extra text."""

    def __init__(self, provider: Literal["openai", "gemini"] = "openai"):
        """
        Initialize LLM matcher.
        
        Args:
            provider: "openai" or "gemini"
            
        Raises:
            ValueError: If provider not available or no API key configured
        """
        self.provider = provider
        self.client = None
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
            
        elif provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
            
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-pro")
            self.model = "gemini-pro"
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")
    
    def analyze_match(
        self,
        job_description: str,
        resume_text: str,
        max_tokens: int = 1000
    ) -> LLMMatchResult:
        """
        Analyze job-resume match using LLM.
        
        Args:
            job_description: Full job description text
            resume_text: Resume text (will be truncated to first 4000 chars)
            max_tokens: Maximum tokens for response
            
        Returns:
            LLMMatchResult with detailed analysis
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If LLM API call fails
        """
        if not job_description.strip():
            raise ValueError("Job description cannot be empty")
        if not resume_text.strip():
            raise ValueError("Resume text cannot be empty")
        
        # Truncate resume to avoid token limits
        resume_excerpt = resume_text[:4000]
        
        # Build prompt
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            jd=job_description,
            resume=resume_excerpt
        )
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt, max_tokens)
            else:
                response = self._call_gemini(prompt)
            
            # Parse JSON response
            result = self._parse_response(response)
            
            return result
        
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")
    
    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API."""
        response = self.client.generate_content(
            f"{self.SYSTEM_PROMPT}\n\n{prompt}",
            generation_config={"temperature": 0.3}
        )
        return response.text
    
    @staticmethod
    def _parse_response(response_text: str) -> LLMMatchResult:
        """
        Parse LLM response into structured result.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            LLMMatchResult with parsed data
        """
        # Try to extract JSON from response
        # Sometimes LLM wraps it in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try direct JSON parsing
            json_str = response_text
        
        data = json.loads(json_str)
        
        return LLMMatchResult(
            match_percentage=int(data.get("match_percentage", 0)),
            missing_keywords=data.get("missing_keywords", []),
            weak_areas=data.get("weak_areas", []),
            strengths=data.get("strengths", []),
            detailed_analysis=data.get("detailed_analysis", ""),
            recommendation=data.get("recommendation", ""),
            top_skills_required=data.get("top_skills_required", []),
            top_skills_matched=data.get("top_skills_matched", [])
        )
    
    def compare_with_tfidf(
        self,
        job_description: str,
        resume_text: str,
        tfidf_score: float
    ) -> Dict:
        """
        Compare LLM analysis with TF-IDF scoring.
        
        Args:
            job_description: Job description text
            resume_text: Resume text
            tfidf_score: TF-IDF similarity score (0-100)
            
        Returns:
            Dictionary with comparison and combined insights
        """
        llm_result = self.analyze_match(job_description, resume_text)
        
        # Calculate agreement
        score_diff = abs(llm_result.match_percentage - tfidf_score)
        
        # Determine which is more reliable
        if score_diff < 10:
            agreement = "High agreement between LLM and TF-IDF analysis"
            reliability = "Very confident in assessment"
        elif score_diff < 20:
            agreement = "Moderate agreement between methods"
            reliability = "Reasonably confident in assessment"
        else:
            agreement = "Low agreement - methods differ significantly"
            reliability = "Recommend manual review"
        
        return {
            'llm_result': llm_result,
            'tfidf_score': round(tfidf_score, 2),
            'llm_score': llm_result.match_percentage,
            'score_difference': round(score_diff, 2),
            'agreement': agreement,
            'reliability': reliability,
            'combined_score': round((llm_result.match_percentage + tfidf_score) / 2, 2),
            'recommendation': self._combine_recommendations(llm_result, tfidf_score)
        }
    
    @staticmethod
    def _combine_recommendations(llm_result: LLMMatchResult, tfidf_score: float) -> str:
        """Generate combined recommendation based on both analyses."""
        combined = (llm_result.match_percentage + tfidf_score) / 2
        
        if combined >= 80:
            return f"✅ Strong match ({combined:.0f}% combined). {llm_result.recommendation}"
        elif combined >= 60:
            return f"✅ Good match ({combined:.0f}% combined). {llm_result.recommendation}"
        elif combined >= 40:
            return f"⚠️ Moderate match ({combined:.0f}% combined). {llm_result.recommendation}"
        else:
            return f"❌ Weak match ({combined:.0f}% combined). {llm_result.recommendation}"


class PromptOptimizer:
    """
    Utility class for optimizing prompts for job matching analysis.
    """
    
    @staticmethod
    def extract_key_skills(job_description: str, max_skills: int = 12) -> List[str]:
        """
        Extract top skills from job description using simple NLP.
        Useful for quick skill identification without LLM.
        
        Args:
            job_description: Job description text
            max_skills: Maximum number of skills to extract
            
        Returns:
            List of extracted skills
        """
        # Common skill patterns
        skill_patterns = [
            r'(?:experience|proficiency|expertise|knowledge|skilled)\s+(?:in|with)\s+([A-Za-z\+\#\s]+?)(?:[,;.]|and)',
            r'(?:required|must)\s+(?:have|know|understand)\s+([A-Za-z\+\#\s]+?)(?:[,;.]|and)',
            r'\*\*([A-Za-z\+\#\s]+?)\*\*',  # Markdown bold
            r'`([A-Za-z\+\#\s]+?)`',  # Inline code
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE)
            skills.update([m.strip() for m in matches if m.strip()])
        
        # Also look for common tech skills
        tech_keywords = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'git', 'ci/cd', 'jenkins', 'gitlab',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'fastapi',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn', 'keras',
            'data science', 'analytics', 'tableau', 'power bi', 'excel', 'statistics',
            'agile', 'scrum', 'kanban', 'jira', 'trello', 'asana',
            'api', 'rest', 'graphql', 'grpc', 'microservices',
            'testing', 'pytest', 'junit', 'unittest', 'rspec', 'jest',
            'linux', 'unix', 'windows', 'macos', 'bash', 'shell', 'powershell',
        ]
        
        job_lower = job_description.lower()
        for keyword in tech_keywords:
            if keyword in job_lower:
                skills.add(keyword.title())
        
        return list(skills)[:max_skills]
    
    @staticmethod
    def create_tailored_prompt(
        job_description: str,
        resume_text: str,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """
        Create a tailored analysis prompt with specific focus areas.
        
        Args:
            job_description: Job description
            resume_text: Resume text
            focus_areas: Specific areas to focus on (e.g., ["leadership", "cloud"])
            
        Returns:
            Formatted prompt string
        """
        focus_text = ""
        if focus_areas:
            focus_text = f"\nSpecific areas to assess: {', '.join(focus_areas)}"
        
        return f"""You are an expert recruiter analyzing a job-resume match.

JOB DESCRIPTION:
{job_description[:3000]}

RESUME EXCERPT:
{resume_text[:3000]}
{focus_text}

Provide JSON analysis with: match_percentage, missing_keywords, weak_areas, strengths, 
top_skills_required, top_skills_matched, detailed_analysis, recommendation."""


def is_llm_available() -> tuple[bool, str]:
    """
    Check which LLM providers are available.
    
    Returns:
        Tuple of (has_available, provider_name)
    """
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        return True, "openai"
    elif GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        return True, "gemini"
    else:
        return False, "none"
