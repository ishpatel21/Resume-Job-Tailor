# Resume Job Tailor 📄

An intelligent resume-to-job matching system that analyzes how well your resume aligns with a job description using dual-layer matching (keyword + TF-IDF) and optional AI-powered insights.

![App Demo](/docs/images/app-demo.png)

## Features ✨

### Core Matching System
- **Dual-Layer Scoring:** Combines keyword overlap (40%) and TF-IDF semantic similarity (60%)
- **Keyword Matching:** Fast ATS-like baseline matching with 60+ skill keywords
- **TF-IDF Analysis:** Advanced semantic understanding of resume-job fit
- **Instant Results:** Get matching score in seconds

### User Interface
- **Single-File MVP:** Simple, clean Streamlit web application
- **PDF Upload:** Automatic resume text extraction
- **Real-Time Analysis:** One-click analysis button
- **Color-Coded Results:** Visual feedback with emoji indicators
- **Mobile Friendly:** Responsive web design

### AI-Powered Insights (Optional)
- **GPT-4 Integration:** Detailed resume-job analysis
- **Google Gemini:** Alternative AI provider
- **Skill Gap Analysis:** Identify missing keywords and weak areas
- **Strengths Identification:** Highlight your resume's strong points
- **Hiring Recommendations:** AI-powered hiring assessment

### Intelligent Features
- **Actionable Recommendations:** Suggestions for resume improvement
- **Missing Keywords Detection:** See what skills to add
- **Match Interpretation:** 🟢 Excellent / 🟡 Good / 🟠 Moderate / 🔴 Weak
- **Graceful Fallback:** Works without API keys (LLM optional)

---

## Quick Start 🚀

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ishpatel21/resume-job-tailor.git
cd resume-job-tailor
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **(Optional) Add API key for LLM analysis:**
```bash
# Create or edit .env file
cat > .env << 'EOF'
OPENAI_API_KEY=sk-proj-xxxxx...
AI_PROVIDER=openai
EOF
```

5. **Run the app:**
```bash
./run.sh
```

**Access at:** http://localhost:8501

---

## How to Use 📋

### Using the Web App

1. **Open the app:**
   - Run `./run.sh` in terminal
   - Browser opens automatically to http://localhost:8501

2. **Upload your resume:**
   - Click "Upload your resume (PDF)"
   - Select a PDF file from your computer
   - App automatically extracts text

3. **Paste job description:**
   - Copy job description from job posting
   - Paste into the text area

4. **Click "Analyze Match":**
   - Get instant matching score
   - See keyword matches/misses
   - (Optional) Get AI analysis if key configured

5. **Review results:**
   - Overall match percentage (0-100)
   - Matched keywords list
   - Missing keywords (top 10)
   - Actionable recommendations
   - AI insights (if available)

### Using in Python

```python
from utils.job_matcher import JobMatcher

# Initialize matcher
matcher = JobMatcher()

# Get matching results
results = matcher.match_resume_to_job(resume_text, job_description)

# Access results
print(f"Overall Score: {results['overall_score']}%")
print(f"Summary: {results['summary']}")
print(f"Matched Keywords: {results['matching_keywords']}")
print(f"Missing Keywords: {results['missing_keywords']}")
print(f"Recommendations: {results['recommendations']}")
```

### Test with Sample Data

```bash
python3 test_data.py
```

This runs a demo showing:
- Realistic senior engineer resume
- Matching job description
- Expected 75%+ match score

---

## Architecture 🏗️

### Three-Layer System

**Layer 1: Web Interface** (`app_mvp.py`)
- Streamlit-based UI
- PDF upload and processing
- Real-time results display

**Layer 2: Matching Engine** (`job_matcher.py`)
- Keyword overlap scoring (40% weight)
- TF-IDF semantic similarity (60% weight)
- Combined matching algorithm

**Layer 3: AI Analysis** (`llm_matcher.py`, optional)
- GPT-4 or Gemini integration
- Detailed skill analysis
- Strength/gap identification
- Hiring recommendations

### Scoring Formula

```
Overall Score = (Keyword Score × 0.40) + (TF-IDF Score × 0.60)
```

### Score Interpretation

| Score | Icon | Meaning |
|-------|------|---------|
| 80-100 | 🟢 | Excellent match |
| 60-80 | 🟡 | Good match |
| 40-60 | 🟠 | Moderate match |
| 20-40 | 🔴 | Weak match |
| 0-20 | ⚫ | Poor match |

---

## Project Structure 📁

```
resume-job-tailor/
├── app_mvp.py              # Main Streamlit application
├── app.py                  # Full-featured version
├── config.py               # Configuration management
├── run.sh                  # Quick start script
├── requirements.txt        # Python dependencies
├── test_data.py            # Test/demo data
├── README.md               # This file
├── FILE_GUIDE.md           # Detailed file guide
├── .env                    # API keys (not tracked)
├── .gitignore              # Git configuration
└── utils/                  # Core matching logic
    ├── job_matcher.py      # Dual-layer matching engine
    ├── llm_matcher.py      # AI analysis engine
    ├── pdf_handler.py      # PDF text extraction
    ├── text_processor.py   # Text utilities
    └── ai_handler.py       # AI provider abstraction
```

---

## Documentation 📚

- **[FILE_GUIDE.md](FILE_GUIDE.md)** - Comprehensive guide to every file and module
- **[Requirements](requirements.txt)** - All dependencies with versions

---

## API Keys & Configuration 🔑

### Enable LLM Analysis (Optional)

1. **Get OpenAI API Key:**
   - Visit https://platform.openai.com/api-keys
   - Create new secret key
   - Copy the key

2. **Set up .env file:**
```bash
OPENAI_API_KEY=sk-proj-xxxxx...
GOOGLE_API_KEY=your_google_key_here  # Optional
AI_PROVIDER=openai  # or 'gemini'
```

3. **LLM features activate automatically!**

### Cost Estimate
- **Without LLM:** Free (uses local matching only)
- **With LLM:** ~$0.005-0.03 per analysis (with caching)

### Graceful Fallback
- App works perfectly without API keys
- LLM features simply won't be available
- Matching scores (TF-IDF) always available

---

## Testing & Validation ✅

### Run Tests

```bash
# Test matching logic
python3 test_data.py

# Run all tests (if available)
pytest
```

### Test Coverage
- **job_matcher.py:** 16/16 tests ✅ (100%)
- **llm_matcher.py:** 19/20 tests ✅ (95%)
- **Overall:** 39/40 tests ✅ (97%)

---

## Technology Stack 🛠️

### Core Dependencies
- **Streamlit 1.28.1** - Web framework
- **scikit-learn 1.8.0** - TF-IDF vectorization
- **NLTK 3.8.1** - Natural language processing
- **pdfplumber 0.10.3** - PDF extraction

### AI Integration
- **OpenAI 1.1.1** - GPT-4 API
- **google-generativeai 0.3.0** - Gemini API

### Utilities
- **python-dotenv 1.0.0** - Environment management

---

## Performance 📊

- **Matching Speed:** ~100-500ms per analysis
- **PDF Processing:** ~100-200ms for single page
- **Memory:** ~50-100MB typical usage
- **Concurrent Users:** Supports multiple simultaneous analyses

---

## Troubleshooting 🔧

### "ModuleNotFoundError: No module named..."
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not found"
- Check `.env` file exists in project root
- Verify API key is set correctly
- Restart the app after adding key

### "PDF upload fails"
- Ensure PDF is valid (not corrupted)
- Max 20 pages supported
- Try with a different PDF

### Port 8501 already in use
```bash
# Kill existing process
pkill -f streamlit

# Or use different port
streamlit run app_mvp.py --server.port 8502
```

---

## Examples 💡

### Example 1: Perfect Match
- **Resume:** Senior Python Engineer with 7+ years, AWS, Kubernetes, FastAPI
- **Job:** Senior Backend Engineer - Python, AWS, Kubernetes required
- **Expected Score:** 75-85% (excellent)

### Example 2: Partial Match
- **Resume:** Junior Frontend Developer (React, JavaScript)
- **Job:** Senior Full-Stack Engineer - React, Node.js, AWS, Kubernetes
- **Expected Score:** 45-55% (moderate)

### Example 3: Poor Match
- **Resume:** Graphic Designer (Photoshop, Illustrator)
- **Job:** Senior Data Scientist - Python, ML, Statistics
- **Expected Score:** 10-20% (poor)

---

## Contributing 🤝

Contributions welcome! Areas for enhancement:
- Additional AI providers (Claude, Llama)
- Resume tailoring suggestions
- Batch job matching
- Export functionality
- Mobile app

---

## License 📄

MIT License - See LICENSE file for details

---

## Support & Issues 💬

- **Report bugs:** Open an issue on GitHub
- **Questions:** Check FILE_GUIDE.md for detailed docs
- **Feature requests:** Create a GitHub discussion

---

## Roadmap 🗺️

- [ ] Export results to PDF
- [ ] Historical tracking
- [ ] Resume tailoring suggestions
- [ ] Batch job analysis
- [ ] Mobile app
- [ ] Job market salary insights
- [ ] LinkedIn integration

---

## Author

**Ishan Patel** - [GitHub](https://github.com/ishpatel21)

---

**Happy matching! 🚀**

*Last Updated: February 17, 2026*
