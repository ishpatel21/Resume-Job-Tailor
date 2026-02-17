"""
Resume ↔ Job Matcher MVP
A single-file Streamlit app combining keyword, TF-IDF, and LLM matching.
"""

import streamlit as st
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Import our utilities from new structure
from processors.pdf_handler import extract_resume_text
from src.matching.job_matcher import JobMatcher, SKLEARN_AVAILABLE
from src.matching.llm_matcher import LLMJobMatcher, is_llm_available, PromptOptimizer

st.set_page_config(
    page_title="Resume ↔ Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SESSION STATE & CONFIG
# ============================================================================

if "matcher" not in st.session_state:
    st.session_state.matcher = JobMatcher() if SKLEARN_AVAILABLE else None

if "llm_matcher" not in st.session_state:
    st.session_state.llm_matcher = None
    st.session_state.llm_provider = None
    llm_available, provider = is_llm_available()
    if llm_available:
        try:
            st.session_state.llm_matcher = LLMJobMatcher(provider=provider)
            st.session_state.llm_provider = provider
        except Exception:
            pass


# ============================================================================
# MAIN UI
# ============================================================================

st.title("📄 Resume ↔ Job Matcher")
st.markdown("*Day 1 MVP: Analyze your job fit instantly*")
st.divider()

# Two-column layout for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Your Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type="pdf",
        key="resume_uploader"
    )
    resume_text = None
    if uploaded_file:
        st.success("✓ Resume loaded")
        try:
            resume_bytes = uploaded_file.read()
            resume_text = extract_resume_text(resume_bytes)
            st.caption(f"{len(resume_text):,} characters extracted")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

with col2:
    st.subheader("📋 Job Description")
    job_desc = st.text_area(
        "Paste the full job description here",
        height=250,
        placeholder="Paste job posting...",
        key="job_input"
    )
    if job_desc:
        st.caption(f"{len(job_desc):,} characters")

st.divider()

# Analysis button
if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if not resume_text:
        st.error("❌ Please upload a resume")
    elif not job_desc:
        st.error("❌ Please paste a job description")
    else:
        with st.spinner("Extracting & analyzing..."):
            # ================================================================
            # ANALYSIS LAYER 1: Keyword + TF-IDF (always available)
            # ================================================================
            
            if st.session_state.matcher and SKLEARN_AVAILABLE:
                tfidf_results = st.session_state.matcher.match_resume_to_job(
                    resume_text, 
                    job_desc
                )
                
                overall_score = tfidf_results['overall_score']
                keyword_score = tfidf_results['keyword_overlap']['score']
                semantic_score = tfidf_results['tfidf_similarity']['score']
                matching_keywords = tfidf_results['matching_keywords'][:15]
                missing_keywords = tfidf_results['missing_keywords'][:10]
                recommendations = tfidf_results['recommendations']
            else:
                # Fallback if scikit-learn not available
                overall_score = 0
                keyword_score = 0
                semantic_score = 0
                matching_keywords = []
                missing_keywords = []
                recommendations = []
            
            # ================================================================
            # ANALYSIS LAYER 2: LLM (if available)
            # ================================================================
            
            llm_result = None
            if st.session_state.llm_matcher:
                try:
                    llm_result = st.session_state.llm_matcher.analyze_match(
                        job_desc,
                        resume_text
                    )
                except Exception as e:
                    st.warning(f"LLM analysis unavailable: {str(e)[:100]}")
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            st.success("✓ Analysis complete")
            st.divider()
            
            # Main score display
            st.subheader("📊 Match Analysis")
            
            col_score1, col_score2, col_score3 = st.columns(3)
            
            with col_score1:
                score_color = "🟢" if overall_score >= 70 else "🟡" if overall_score >= 50 else "🔴"
                st.metric(
                    "Overall Score",
                    f"{overall_score:.0f}%",
                    delta=score_color
                )
            
            with col_score2:
                st.metric("Keyword Match", f"{keyword_score:.0f}%")
            
            with col_score3:
                st.metric("Semantic Match", f"{semantic_score:.0f}%")
            
            # If LLM available, show AI score
            if llm_result:
                st.divider()
                st.subheader("🤖 AI-Powered Analysis")
                
                col_llm1, col_llm2 = st.columns(2)
                
                with col_llm1:
                    ai_color = "🟢" if llm_result.match_percentage >= 70 else "🟡" if llm_result.match_percentage >= 50 else "🔴"
                    st.metric(
                        "AI Match Score",
                        f"{llm_result.match_percentage}%",
                        delta=ai_color
                    )
                
                with col_llm2:
                    agreement = "🟢 HIGH" if abs(overall_score - llm_result.match_percentage) < 10 else "🟡 MEDIUM" if abs(overall_score - llm_result.match_percentage) < 20 else "🔴 LOW"
                    st.metric("Method Agreement", agreement)
                
                # Recommendation
                st.info(f"**💼 Recommendation:** {llm_result.recommendation}")
                
                # Detailed analysis
                st.markdown("#### Detailed Analysis")
                st.write(llm_result.detailed_analysis)
                
                # Strengths
                if llm_result.strengths:
                    st.markdown("#### ✅ Your Strengths")
                    for strength in llm_result.strengths:
                        st.success(f"✓ {strength}")
                
                # Weak areas
                if llm_result.weak_areas:
                    st.markdown("#### ⚠️ Areas to Improve")
                    for area in llm_result.weak_areas:
                        st.warning(f"→ {area}")
                
                # Missing keywords
                if llm_result.missing_keywords:
                    st.markdown("#### 🎯 Missing / Weak Keywords")
                    missing_text = ", ".join(llm_result.missing_keywords)
                    st.error(missing_text)
                
                # Skills breakdown
                col_skills1, col_skills2 = st.columns(2)
                
                with col_skills1:
                    st.markdown("#### Top Skills Required")
                    for skill in llm_result.top_skills_required[:6]:
                        st.caption(f"• {skill}")
                
                with col_skills2:
                    st.markdown("#### Your Matching Skills")
                    if llm_result.top_skills_matched:
                        for skill in llm_result.top_skills_matched:
                            st.caption(f"✓ {skill}")
                    else:
                        st.caption("(None detected)")
                
                st.caption(f"🤖 Analysis by: {st.session_state.llm_provider.upper()}")
            
            else:
                # No LLM available - show TF-IDF results
                st.divider()
                st.subheader("✅ Keyword & Semantic Analysis")
                
                if matching_keywords:
                    st.markdown("#### Matching Keywords")
                    keywords_text = ", ".join(matching_keywords)
                    st.success(f"Found in resume: {keywords_text}")
                else:
                    st.warning("No matching keywords found")
                
                if missing_keywords:
                    st.markdown("#### Missing / Weak Keywords")
                    missing_text = ", ".join(missing_keywords)
                    st.error(missing_text)
                
                if recommendations:
                    st.markdown("#### Recommendations")
                    for rec in recommendations:
                        st.info(f"→ {rec}")
                
                # Hint to enable LLM
                st.info(
                    "💡 **Tip:** Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` environment "
                    "variable to see AI-powered analysis with detailed insights."
                )

st.divider()

# Footer
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    if st.button("📖 View Documentation", use_container_width=True):
        st.info("See `LLM_QUICK_START.md` for setup and `LLM_MATCHING_GUIDE.md` for details.")

with col_footer2:
    if st.button("🧪 Run Tests", use_container_width=True):
        st.info("`pytest test_llm_matcher.py -v`")

with col_footer3:
    if st.button("🔧 System Status", use_container_width=True):
        st.info(f"""
        **System Status:**
        - TF-IDF: {'✅ Ready' if SKLEARN_AVAILABLE else '❌ Not available'}
        - LLM: {'✅ Ready' if st.session_state.llm_matcher else '⚠️ API key missing'}
        - Provider: {st.session_state.llm_provider.upper() if st.session_state.llm_provider else 'None'}
        """)

st.caption("Resume Job Tailor • Day 1 MVP • Built with ❤️")
