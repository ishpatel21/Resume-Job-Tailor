"""
PDF handling utilities for extracting text and tables from resumes.

Provides robust, production-ready PDF extraction with:
- Support for both file paths and bytes input
- Comprehensive text cleaning and normalization
- Table extraction and formatting
- Error handling and validation
- Page limit enforcement
"""
import pdfplumber
import re
import io
from pathlib import Path
from config import PDF_MAX_PAGES
from typing import Union, List, Dict, Optional, Tuple


# Text cleaning patterns
_EXTRA_SPACES = re.compile(r'\s+')
_SPECIAL_CHARS = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]')
_BULLET_POINTS = re.compile(r'^[\s•◦▪■□○◆★]\s+', re.MULTILINE)


def _validate_pdf_input(pdf_input: Union[str, bytes]) -> Union[str, io.BytesIO]:
    """
    Validate and normalize PDF input.
    
    Args:
        pdf_input: Either a file path (str) or PDF bytes
        
    Returns:
        Either a file path (str) or BytesIO object
        
    Raises:
        ValueError: If input is invalid or file doesn't exist
    """
    if isinstance(pdf_input, str):
        path = Path(pdf_input)
        if not path.exists():
            raise ValueError(f"PDF file not found: {pdf_input}")
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_input}")
        return str(path)
    elif isinstance(pdf_input, bytes):
        if len(pdf_input) == 0:
            raise ValueError("PDF bytes cannot be empty")
        # Verify PDF signature
        if not pdf_input.startswith(b'%PDF'):
            raise ValueError("Invalid PDF format: file does not start with PDF signature")
        return io.BytesIO(pdf_input)
    else:
        raise ValueError(f"Expected str or bytes, got {type(pdf_input)}")


def _clean_text(text: str) -> str:
    """
    Comprehensively clean extracted text.
    
    Handles:
    - Control characters removal
    - Extra whitespace normalization
    - Newline preservation for structure
    - Empty line cleanup
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove control characters
    text = _SPECIAL_CHARS.sub('', text)
    
    # Normalize spaces within lines (preserve newlines)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove extra spaces from this line
        line = _EXTRA_SPACES.sub(' ', line).strip()
        
        # Remove leading bullet points but preserve content
        line = _BULLET_POINTS.sub('', line)
        
        if line:  # Only add non-empty lines
            cleaned_lines.append(line)
    
    # Join with newlines and normalize final result
    result = '\n'.join(cleaned_lines)
    
    # Final pass: normalize multiple newlines to max 2
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def _extract_text_from_page(page) -> str:
    """
    Extract text from a single page with fallback methods.
    
    Tries multiple extraction methods for robustness:
    1. Standard extract_text()
    2. extract_text() with layout mode
    3. extract_text() with OCR settings
    
    Args:
        page: A pdfplumber page object
        
    Returns:
        Extracted text from the page
    """
    # Try standard extraction
    text = page.extract_text()
    if text and text.strip():
        return text
    
    # Fallback: try layout mode for better structure
    try:
        text = page.extract_text(layout=True)
        if text and text.strip():
            return text
    except:
        pass
    
    # Final fallback: extract from objects if text extraction fails
    try:
        chars = page.chars
        text = "".join([char['text'] for char in chars if 'text' in char])
        if text and text.strip():
            return text
    except:
        pass
    
    return ""


def extract_resume_text(pdf_input: Union[str, bytes]) -> str:
    """
    Extract and clean text from a resume PDF.
    
    Main extraction function supporting both file paths and bytes.
    Handles errors gracefully and applies comprehensive text cleaning.
    
    Args:
        pdf_input: Either a file path (str) or PDF bytes
        
    Returns:
        Cleaned, normalized text from the resume
        
    Raises:
        ValueError: If PDF is invalid or cannot be read
        
    Example:
        # From file
        text = extract_resume_text("/path/to/resume.pdf")
        
        # From Streamlit upload
        uploaded_file = st.file_uploader("Upload PDF")
        if uploaded_file:
            text = extract_resume_text(uploaded_file.read())
    """
    try:
        # Validate input
        pdf_source = _validate_pdf_input(pdf_input)
        
        text_parts = []
        
        with pdfplumber.open(pdf_source) as pdf:
            # Validate PDF has pages
            if not pdf.pages:
                raise ValueError("PDF has no pages")
            
            # Extract from pages up to limit
            page_limit = min(PDF_MAX_PAGES, len(pdf.pages))
            
            for page_num, page in enumerate(pdf.pages[:page_limit], 1):
                try:
                    page_text = _extract_text_from_page(page)
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    # Log but continue with other pages
                    print(f"Warning: Error extracting page {page_num}: {e}")
                    continue
        
        # Combine all text and clean
        combined_text = "\n".join(text_parts)
        cleaned_text = _clean_text(combined_text)
        
        if not cleaned_text:
            raise ValueError("No text could be extracted from PDF")
        
        return cleaned_text
        
    except ValueError:
        # Re-raise ValueError exceptions (validation errors)
        raise
    except Exception as e:
        # Catch any other PDF reading errors
        raise ValueError(f"Failed to extract text from PDF: {e}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    
    Wrapper around extract_resume_text for file paths.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    return extract_resume_text(pdf_path)


def extract_tables_from_pdf(pdf_input: Union[str, bytes]) -> List[List[List[str]]]:
    """
    Extract tables from PDF using pdfplumber.
    
    Supports both file paths and bytes input.
    Returns structured table data suitable for processing.
    
    Args:
        pdf_input: Either a file path (str) or PDF bytes
        
    Returns:
        List of tables, where each table is a list of rows
    """
    try:
        pdf_source = _validate_pdf_input(pdf_input)
        tables = []
        
        with pdfplumber.open(pdf_source) as pdf:
            page_limit = min(PDF_MAX_PAGES, len(pdf.pages))
            
            for page_num, page in enumerate(pdf.pages[:page_limit], 1):
                try:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            # Convert table to list of lists of strings
                            formatted_table = [
                                [str(cell) if cell else "" for cell in row]
                                for row in table
                            ]
                            tables.append(formatted_table)
                except Exception as e:
                    print(f"Warning: Error extracting tables from page {page_num}: {e}")
                    continue
        
        return tables
        
    except Exception as e:
        print(f"Warning: Could not extract tables: {e}")
        return []


def format_tables_as_text(tables: List[List[List[str]]]) -> str:
    """
    Format extracted tables as readable text.
    
    Converts table data into formatted text representation
    suitable for inclusion in resume text.
    
    Args:
        tables: List of tables from extract_tables_from_pdf()
        
    Returns:
        Formatted text representation of tables
    """
    if not tables:
        return ""
    
    formatted = []
    
    for table_idx, table in enumerate(tables, 1):
        formatted.append(f"--- Table {table_idx} ---")
        
        for row in table:
            # Join cells with tab separation
            row_text = " | ".join(str(cell).strip() for cell in row)
            formatted.append(row_text)
        
        formatted.append("")  # Blank line between tables
    
    return "\n".join(formatted)


def extract_resume_text_with_tables(pdf_input: Union[str, bytes]) -> Tuple[str, List[List[List[str]]]]:
    """
    Extract both text and tables from a resume PDF.
    
    Comprehensive extraction including both main text and tabular data.
    Useful when resume contains skill matrices, job history tables, etc.
    
    Args:
        pdf_input: Either a file path (str) or PDF bytes
        
    Returns:
        Tuple of (cleaned_text, tables_list)
        
    Example:
        text, tables = extract_resume_text_with_tables(pdf_bytes)
        if tables:
            formatted_tables = format_tables_as_text(tables)
            complete_text = text + "\n" + formatted_tables
    """
    text = extract_resume_text(pdf_input)
    tables = extract_tables_from_pdf(pdf_input)
    return text, tables


def get_pdf_metadata(pdf_input: Union[str, bytes]) -> Dict[str, any]:
    """
    Extract metadata from PDF.
    
    Args:
        pdf_input: Either a file path (str) or PDF bytes
        
    Returns:
        Dictionary with PDF metadata
    """
    try:
        pdf_source = _validate_pdf_input(pdf_input)
        
        with pdfplumber.open(pdf_source) as pdf:
            metadata = {
                "page_count": len(pdf.pages),
                "metadata": pdf.metadata or {},
                "has_tables": False,
                "estimated_text_length": 0
            }
            
            # Quick check for tables and text
            for page in pdf.pages[:PDF_MAX_PAGES]:
                if page.extract_tables():
                    metadata["has_tables"] = True
                text = page.extract_text()
                if text:
                    metadata["estimated_text_length"] += len(text)
            
            return metadata
            
    except Exception as e:
        print(f"Warning: Could not extract metadata: {e}")
        return {
            "page_count": 0,
            "metadata": {},
            "has_tables": False,
            "estimated_text_length": 0
        }


def save_pdf(content: str, output_path: str) -> None:
    """
    Save content to a PDF file (placeholder for future implementation).
    
    Args:
        content: Text content to save
        output_path: Path to save the PDF
    """
    # This will be implemented when adding PDF export functionality
    pass
