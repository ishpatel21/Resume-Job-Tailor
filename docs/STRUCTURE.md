# Project Structure Guide

## Overview

Resume Job Tailor uses a modern, well-organized package structure for scalability and maintainability.

```
resume-job-tailor/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── app_mvp.py               # Main Streamlit application
│   ├── config.py                # Configuration management
│   └── matching/                # Matching algorithms
│       ├── __init__.py
│       ├── job_matcher.py       # Dual-layer matching (keyword + TF-IDF)
│       └── llm_matcher.py       # AI-powered analysis (GPT-4/Gemini)
│
├── processors/                   # Data processing utilities
│   ├── __init__.py
│   ├── pdf_handler.py           # PDF text extraction
│   ├── text_processor.py        # Text normalization & processing
│   └── ai_handler.py            # AI provider abstraction
│
├── tests/                        # Test data & validation
│   ├── __init__.py
│   └── test_data.py             # Demo resume/job data with 75%+ match
│
├── scripts/                      # Utility scripts
│   └── run.sh                   # Alternative run script location
│
├── docs/                         # Documentation
│   ├── README.md                # Main project documentation
│   └── FILE_GUIDE.md            # Detailed file reference
│
├── app.py                        # Full-featured version (reference)
├── run.sh                        # Root-level quick start script
├── requirements.txt              # Python dependencies
├── .env                         # API keys (not tracked)
├── .gitignore                   # Git configuration
└── .venv/                       # Virtual environment (not tracked)
```

---

## Directory Breakdown

### `src/` - Main Application

**Purpose:** Core application and matching logic

**Files:**
- **`app_mvp.py`** - Streamlit web interface
  - Handles UI/UX
  - PDF upload
  - Real-time analysis
  - Results display

- **`config.py`** - Configuration management
  - Loads environment variables
  - API key management
  - Settings initialization

- **`matching/`** - Matching algorithms
  - `job_matcher.py` - Dual-layer matching engine
  - `llm_matcher.py` - AI-powered analysis

### `processors/` - Data Processing

**Purpose:** Text and data processing utilities

**Files:**
- **`pdf_handler.py`** - PDF text extraction
  - Extracts text from PDF resumes
  - Multi-page support
  - Error handling

- **`text_processor.py`** - Text processing utilities
  - Text normalization
  - Keyword extraction
  - Tokenization
  - Stopword removal

- **`ai_handler.py`** - AI provider abstraction
  - OpenAI integration
  - Gemini integration
  - Error handling & fallbacks

### `tests/` - Testing & Demo Data

**Purpose:** Test data and validation

**Files:**
- **`test_data.py`** - High-match demo data
  - Realistic resume example
  - Matching job description
  - 75%+ expected match

**Run tests:**
```bash
python3 tests/test_data.py
```

### `scripts/` - Utility Scripts

**Purpose:** Helper scripts for development

**Files:**
- **`run.sh`** - Alternative run script location
  - Can be used from anywhere
  - Calls root-level `run.sh`

### `docs/` - Documentation

**Purpose:** Project documentation

**Files:**
- **`README.md`** - Main documentation
  - Features overview
  - Installation & setup
  - Usage guide
  - Architecture explanation

- **`FILE_GUIDE.md`** - Detailed file reference
  - Every file explained
  - Function documentation
  - Usage examples

---

## Imports by Location

### From `src/app_mvp.py`

```python
# Process PDFs
from processors.pdf_handler import extract_resume_text

# Matching
from src.matching.job_matcher import JobMatcher, SKLEARN_AVAILABLE
from src.matching.llm_matcher import LLMJobMatcher, is_llm_available

# Configuration
from src.config import get_config
```

### From anywhere in the project

```python
# Matching
from src.matching import JobMatcher, LLMJobMatcher

# Processing
from processors import extract_resume_text, normalize_text

# Text utilities
from processors.text_processor import extract_keywords, tokenize
```

---

## Running the Application

### Quick Start (Root Directory)
```bash
./run.sh
```

### From Scripts Directory
```bash
./scripts/run.sh
```

### Direct with Python
```bash
.venv/bin/streamlit run src/app_mvp.py
```

### On Different Port
```bash
.venv/bin/streamlit run src/app_mvp.py --server.port 8502
```

---

## Testing

### Run Demo with High-Match Resume
```bash
python3 tests/test_data.py
```

**Expected Output:**
```
📊 Overall Match Score: 75.44%
📍 Summary: 🟡 Good match - Your resume covers most requirements

Keyword Overlap: 100.0%
  ✓ Matched: 30 keywords
  ✗ Missing: 0 keywords

TF-IDF Similarity: 59.06%
  Interpretation: Moderate match
```

---

## Python Import Paths

### Relative Imports (within packages)

```python
# In src/matching/job_matcher.py
from ..config import get_config
```

### Absolute Imports (recommended)

```python
# From anywhere
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching import JobMatcher
from processors import extract_resume_text
```

---

## Adding New Modules

### To add a new matching algorithm:
```
src/matching/new_matcher.py
```
Then update `src/matching/__init__.py`:
```python
from .new_matcher import NewMatcher
__all__ = [..., 'NewMatcher']
```

### To add a new processor:
```
processors/new_processor.py
```
Then update `processors/__init__.py`:
```python
from .new_processor import new_function
__all__ = [..., 'new_function']
```

### To add tests:
```
tests/test_new_feature.py
```

---

## Benefits of This Structure

✅ **Modularity** - Easy to find and modify code
✅ **Scalability** - Easy to add new features
✅ **Maintainability** - Clear organization
✅ **Testing** - Dedicated test directory
✅ **Documentation** - Centralized docs folder
✅ **Production-Ready** - Professional package structure
✅ **IDE Support** - Better autocomplete and navigation
✅ **Import Management** - Clean, organized imports

---

## File Size Reference

| Directory | Purpose | Size |
|-----------|---------|------|
| src/ | Main application | ~12 KB |
| src/matching/ | Matching logic | ~800 lines |
| processors/ | Data processing | ~500 lines |
| tests/ | Test data | ~150 lines |
| docs/ | Documentation | ~2000 lines |
| scripts/ | Helper scripts | ~10 lines |

---

## Configuration Files

### Root Level
- `run.sh` - Quick start script (symlink concept)
- `requirements.txt` - Python dependencies
- `.env` - API keys (not tracked)
- `.gitignore` - Git configuration
- `app.py` - Legacy full-featured version

### In Subdirectories
- `src/config.py` - Application configuration
- `src/__init__.py` - Package initialization
- `processors/__init__.py` - Processor module exports
- `src/matching/__init__.py` - Matching module exports

---

## Migration Guide (if upgrading from flat structure)

**Old Structure:**
```
├── app_mvp.py
├── utils/job_matcher.py
└── utils/pdf_handler.py
```

**New Structure:**
```
├── src/app_mvp.py
├── src/matching/job_matcher.py
└── processors/pdf_handler.py
```

**Import Changes:**
```python
# Old
from utils.job_matcher import JobMatcher

# New
from src.matching.job_matcher import JobMatcher
# or
from src.matching import JobMatcher
```

---

## Next Steps

1. **Run the app:** `./run.sh`
2. **View docs:** `cat docs/README.md`
3. **Explore code:** Check specific modules in `src/` and `processors/`
4. **Add features:** Follow the structure guidelines above

---

**This structure is designed to scale as the project grows! 🚀**
