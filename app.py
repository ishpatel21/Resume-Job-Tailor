"""
Main Streamlit application for Resume Job Tailor.
Allows users to upload resume and job description for AI-powered customization.
"""
import streamlit as st
from config import AI_PROVIDER, UPLOAD_DIR, OUTPUT_DIR
from utils.pdf_handler import (
    extract_resume_text, 
    get_pdf_metadata,
    extract_resume_text_with_tables,
    format_tables_as_text
)
from utils.job_matcher import JobMatcher, SKLEARN_AVAILABLE
from utils.llm_matcher import LLMJobMatcher, is_llm_available, OPENAI_AVAILABLE, GEMINI_AVAILABLE
import os

st.set_page_config(
    page_title="Resume Job Tailor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Resume Job Tailor")
st.markdown(
    "Customize your resume to match job descriptions using AI"
)

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.info(
    f"Current AI Provider: **{AI_PROVIDER.upper()}**\n\n"
    "Make sure to set up your API keys in `.env`"
)

# Session state for extracted resume text
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "resume_metadata" not in st.session_state:
    st.session_state.resume_metadata = None
if "job_matcher" not in st.session_state:
    st.session_state.job_matcher = None
    if SKLEARN_AVAILABLE:
        try:
            st.session_state.job_matcher = JobMatcher()
        except Exception as e:
            st.warning(f"⚠️ Job matching requires scikit-learn: {e}")

if "llm_matcher" not in st.session_state:
    st.session_state.llm_matcher = None
    llm_available, provider = is_llm_available()
    if llm_available:
        try:
            st.session_state.llm_matcher = LLMJobMatcher(provider=provider)
            st.session_state.llm_provider = provider
        except Exception as e:
            st.session_state.llm_matcher = None

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Upload Resume")
    resume_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        key="resume_uploader"
    )
    
    if resume_file:
        st.success("✓ Resume uploaded")
        
        # Extract PDF metadata
        try:
            with st.spinner("Reading PDF metadata..."):
                metadata = get_pdf_metadata(resume_file.read())
                st.session_state.resume_metadata = metadata
                
                col_meta1, col_meta2 = st.columns(2)
                with col_meta1:
                    st.metric("Pages", metadata["page_count"])
                with col_meta2:
                    st.metric("Est. Text Length", f"{metadata['estimated_text_length']:,} chars")
                
                if metadata["has_tables"]:
                    st.info("📊 Document contains tables")
        except Exception as e:
            st.error(f"Could not read PDF metadata: {e}")
            resume_file = None

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="Paste job description text here..."
    )
    if job_description:
        st.success("✓ Job description entered")
        st.caption(f"Characters: {len(job_description)}")

# Processing section
if st.button("🚀 Analyze & Match", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload a resume")
    elif not job_description:
        st.error("Please enter a job description")
    else:
        # Extract resume text
        with st.spinner("Extracting resume text..."):
            try:
                # Get resume text and tables
                pdf_bytes = resume_file.read()
                resume_text, tables = extract_resume_text_with_tables(pdf_bytes)
                
                # Store in session state
                st.session_state.resume_text = resume_text
                
                st.success("✓ Resume extracted successfully")
                
            except ValueError as e:
                st.error(f"❌ Error extracting resume: {e}")
                resume_text = None
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                resume_text = None
        
        if resume_text:
            # Display extraction preview
            with st.expander("📄 Extracted Resume Text (Preview)"):
                preview = resume_text[:500] + ("..." if len(resume_text) > 500 else "")
                st.text_area(
                    "Resume Content",
                    value=preview,
                    height=150,
                    disabled=True
                )
                st.caption(f"Total characters: {len(resume_text)}")
            
            if tables:
                with st.expander("📊 Extracted Tables"):
                    tables_text = format_tables_as_text(tables)
                    st.text(tables_text)
            
            # Perform job matching analysis
            st.subheader("🎯 Job Match Analysis")
            
            if not SKLEARN_AVAILABLE:
                st.warning("⚠️ scikit-learn not installed. Install with: pip install scikit-learn")
            else:
                with st.spinner("Analyzing resume-job fit..."):
                    try:
                        matcher = st.session_state.job_matcher or JobMatcher()
                        
                        # Perform matching
                        results = matcher.match_resume_to_job(resume_text, job_description)
                        
                        # Display overall score
                        st.markdown("### Overall Match Score")
                        col_score1, col_score2, col_score3 = st.columns(3)
                        
                        with col_score1:
                            overall = results['overall_score']
                            st.metric(
                                "Overall Score",
                                f"{overall:.1f}%",
                                delta=f"{'✓' if overall >= 60 else '✗'}"
                            )
                        
                        with col_score2:
                            keyword_score = results['keyword_overlap']['score']
                            st.metric("Keyword Overlap", f"{keyword_score:.1f}%")
                        
                        with col_score3:
                            tfidf_score = results['tfidf_similarity']['score']
                            st.metric("Semantic Match", f"{tfidf_score:.1f}%")
                        
                        # Summary
                        st.markdown(f"**{results['summary']}**")
                        
                        # Display matching keywords
                        st.markdown("### 📌 Matching Keywords")
                        if results['matching_keywords']:
                            matching_text = ", ".join(results['matching_keywords'][:20])
                            st.success(f"Found in resume: {matching_text}")
                        else:
                            st.warning("No matching keywords found")
                        
                        # Display missing keywords
                        st.markdown("### 🔍 Missing Keywords (Top 10)")
                        if results['missing_keywords']:
                            missing_cols = st.columns(2)
                            for idx, keyword in enumerate(results['missing_keywords']):
                                with missing_cols[idx % 2]:
                                    st.caption(f"❌ {keyword}")
                        else:
                            st.success("All keywords present!")
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        for i, rec in enumerate(results['recommendations'], 1):
                            st.info(f"{i}. {rec}")
                        
                        # Detailed breakdown
                        st.markdown("### 📊 Detailed Breakdown")
                        col_detail1, col_detail2 = st.columns(2)
                        
                        with col_detail1:
                            st.markdown("**Keyword Overlap Method**")
                            st.markdown(f"""
                            - **Score:** {results['keyword_overlap']['score']:.1f}%
                            - **Matched:** {len(results['keyword_overlap']['matched_keywords'])} of {results['keyword_overlap']['total_keywords']}
                            - **Method:** Simple keyword frequency matching (ATS-like)
                            """)
                        
                        with col_detail2:
                            st.markdown("**TF-IDF Similarity Method**")
                            st.markdown(f"""
                            - **Score:** {results['tfidf_similarity']['score']:.1f}%
                            - **Interpretation:** {results['tfidf_similarity']['interpretation']}
                            - **Method:** Semantic similarity using machine learning
                            """)
                        
                        # Additional statistics
                        st.markdown("### 📈 Statistics")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            word_count = len(resume_text.split())
                            st.metric("Resume Words", word_count)
                        
                        with col_stat2:
                            job_word_count = len(job_description.split())
                            st.metric("Job Description Words", job_word_count)
                        
                        with col_stat3:
                            st.metric("Tables Found", len(tables))
                        
                        # LLM-based analysis (if available)
                        st.markdown("---")
                        st.markdown("### 🤖 AI-Powered Analysis (LLM)")
                        
                        if st.session_state.llm_matcher is not None:
                            with st.spinner("Running AI analysis..."):
                                try:
                                    llm_result = st.session_state.llm_matcher.analyze_match(
                                        job_description, 
                                        resume_text
                                    )
                                    
                                    # Display LLM results in columns
                                    col_llm1, col_llm2 = st.columns(2)
                                    
                                    with col_llm1:
                                        st.metric(
                                            "AI Match Score",
                                            f"{llm_result.match_percentage}%",
                                            delta=f"{'✓' if llm_result.match_percentage >= 70 else '⚠'}"
                                        )
                                    
                                    with col_llm2:
                                        tfidf_score = results['overall_score']
                                        st.metric(
                                            "Algorithm Score",
                                            f"{tfidf_score:.0f}%",
                                            delta=f"{'✓' if tfidf_score >= 60 else '⚠'}"
                                        )
                                    
                                    # Comparison
                                    comparison = st.session_state.llm_matcher.compare_with_tfidf(
                                        job_description,
                                        resume_text,
                                        results['overall_score'] / 100
                                    )
                                    
                                    col_comp1, col_comp2 = st.columns(2)
                                    with col_comp1:
                                        agreement_emoji = "🟢" if comparison['agreement_level'] == 'high' else "🟡" if comparison['agreement_level'] == 'medium' else "🔴"
                                        st.info(f"{agreement_emoji} Agreement: **{comparison['agreement_level'].upper()}**")
                                    with col_comp2:
                                        st.info(f"📊 Confidence: **{comparison['reliability_score']:.0f}%**")
                                    
                                    # AI Recommendation
                                    st.markdown("#### 💼 AI Recommendation")
                                    st.success(f"**{llm_result.recommendation}**")
                                    
                                    # Detailed Analysis
                                    st.markdown("#### 📝 Detailed Analysis")
                                    st.write(llm_result.detailed_analysis)
                                    
                                    # Top Skills Required vs Matched
                                    col_skills1, col_skills2 = st.columns(2)
                                    
                                    with col_skills1:
                                        st.markdown("#### ⭐ Top Skills Required")
                                        for skill in llm_result.top_skills_required[:6]:
                                            st.caption(f"• {skill}")
                                    
                                    with col_skills2:
                                        st.markdown("#### ✅ Skills You Have")
                                        if llm_result.top_skills_matched:
                                            for skill in llm_result.top_skills_matched:
                                                st.caption(f"✓ {skill}")
                                        else:
                                            st.caption("(None detected)")
                                    
                                    # Strengths
                                    st.markdown("#### 💪 Your Strengths")
                                    for strength in llm_result.strengths:
                                        st.success(f"✓ {strength}")
                                    
                                    # Weak Areas
                                    if llm_result.weak_areas:
                                        st.markdown("#### 📉 Areas for Improvement")
                                        for area in llm_result.weak_areas:
                                            st.warning(f"→ {area}")
                                    
                                    # Missing Keywords
                                    if llm_result.missing_keywords:
                                        st.markdown("#### 🎯 Missing Skills to Highlight")
                                        missing_cols = st.columns(2)
                                        for idx, keyword in enumerate(llm_result.missing_keywords):
                                            with missing_cols[idx % 2]:
                                                st.caption(f"❌ {keyword}")
                                    
                                    # Provider info
                                    st.caption(f"🤖 Analysis by: {st.session_state.llm_provider.upper()}")
                                    
                                except Exception as e:
                                    st.error(f"❌ AI analysis failed: {str(e)}")
                                    st.info("Falling back to algorithm-only analysis")
                        else:
                            st.info(
                                "💡 AI analysis not available. Set OPENAI_API_KEY or GOOGLE_API_KEY to enable.\n\n"
                                "Get free API keys:\n"
                                "- OpenAI: https://platform.openai.com/api-keys\n"
                                "- Google Gemini: https://makersuite.google.com/app/apikey"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                
                st.info("✨ AI tailoring and customization coming soon!")

st.divider()
st.caption("Resume Job Tailor • Powered by AI")
