import os
import re
import zipfile
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Import DB and config
import config
from core.db import BaseVectorStore

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Splits a raw string of text into logical chunks of roughly 'chunk_size' characters,
    with an overlap of 'chunk_overlap' to prevent loss of context across splits.
    Splits first on paragraph double-newlines, then line breaks, then sentence periods, then spaces.
    """
    if len(text) <= chunk_size:
        return [text.strip()]
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        # Define default end boundary
        end = start + chunk_size
        
        # If we are near the end of the text, take the remaining slice
        if end >= text_len:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
            
        # Search backwards inside the overlap range for a logical split point
        search_start = max(start, end - chunk_overlap)
        substring = text[search_start:end]
        
        split_idx = -1
        # Try paragraph break
        split_idx = substring.rfind('\n\n')
        
        if split_idx == -1:
            # Try single line break
            split_idx = substring.rfind('\n')
            
        if split_idx == -1:
            # Try sentence boundary
            split_idx = substring.rfind('. ')
            if split_idx != -1:
                split_idx += 1  # Keep the period in the current chunk
                
        if split_idx == -1:
            # Try space boundary
            split_idx = substring.rfind(' ')
            
        # Calculate actual index to slice
        if split_idx != -1:
            actual_end = search_start + split_idx
        else:
            # Hard split if no standard separator is found
            actual_end = end
            
        chunk = text[start:actual_end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Calculate next start point, guaranteeing forward progress
        start = max(start + 1, actual_end - chunk_overlap)
        
    return chunks


def parse_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF file page-by-page, returning a list of dictionaries containing
    the page text and page metadata (source file and page number).
    """
    pages_data = []
    file_name = Path(file_path).name
    
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Clean up consecutive whitespaces
                cleaned_text = re.sub(r'[ \t]+', ' ', text)
                pages_data.append({
                    "text": cleaned_text.strip(),
                    "metadata": {
                        "source": file_name,
                        "page": i + 1,
                        "type": "pdf"
                    }
                })
    except Exception as e:
        print(f"Error parsing PDF file {file_path}: {e}")
        
    return pages_data


def parse_epub(file_path: str) -> list[dict]:
    """
    Parses EPUB ebook format by unzipping it in-memory, extracting HTML documents,
    and retrieving cleaned text from each section.
    """
    sections_data = []
    file_name = Path(file_path).name
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List all documents inside EPUB
            all_files = zip_ref.namelist()
            # Identify reading material (.html, .xhtml)
            html_files = [f for f in all_files if f.endswith(('.html', '.xhtml', '.htm'))]
            html_files.sort()  # Maintain alphabetical/reading sequence
            
            for index, file in enumerate(html_files):
                try:
                    content = zip_ref.read(file)
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Decompose scripts, styles, etc.
                    for s in soup(["script", "style", "meta", "link"]):
                        s.decompose()
                        
                    # Extract heading or section title
                    title_tag = soup.find(['h1', 'h2', 'title'])
                    section_name = title_tag.get_text().strip() if title_tag else f"Section {index + 1}"
                    
                    text = soup.get_text(separator=' ')
                    # Clean spacing and newlines
                    cleaned_text = re.sub(r'\s+', ' ', text).strip()
                    
                    if cleaned_text:
                        sections_data.append({
                            "text": cleaned_text,
                            "metadata": {
                                "source": file_name,
                                "section": section_name,
                                "type": "epub"
                            }
                        })
                except Exception as ex:
                    print(f"Error parsing EPUB file internal segment {file}: {ex}")
    except Exception as e:
        print(f"Error opening EPUB file {file_path}: {e}")
        
    return sections_data


def parse_txt(file_path: str) -> list[dict]:
    """
    Reads a raw text file, returning a single block wrapped in dictionary structure.
    """
    file_name = Path(file_path).name
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            if text.strip():
                return [{
                    "text": text.strip(),
                    "metadata": {
                        "source": file_name,
                        "type": "txt"
                    }
                }]
    except Exception as e:
        print(f"Error reading TXT file {file_path}: {e}")
    return []


def scrape_url(url: str) -> list[dict]:
    """
    Scrapes a web page, strips boilerplate scripts and styling,
    and returns sanitized text content with URL source metadata.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Decompose elements
        for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
            s.decompose()
            
        title = soup.title.string.strip() if soup.title else url
        
        # Extract main text
        text = soup.get_text(separator=' ')
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        if cleaned_text:
            return [{
                "text": cleaned_text,
                "metadata": {
                    "source": url,
                    "title": title,
                    "type": "web"
                }
            }]
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
    return []


# ==============================================================================
# Pipeline Executors
# ==============================================================================
def ingest_file(file_path: str, vector_store: BaseVectorStore) -> int:
    """
    Parses a local file (PDF, EPUB, TXT), chunks its contents,
    generates embeddings, and saves it to the vector database.
    Returns the total count of vector chunks indexed.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    # 1. Parse File Content
    if ext == '.pdf':
        parsed_blocks = parse_pdf(file_path)
    elif ext == '.epub':
        parsed_blocks = parse_epub(file_path)
    elif ext in ('.txt', '.md'):
        parsed_blocks = parse_txt(file_path)
    else:
        print(f"Unsupported file format: {ext}")
        return 0
        
    if not parsed_blocks:
        return 0
        
    # 2. Split and prepare chunks
    final_texts = []
    final_metadatas = []
    
    for block in parsed_blocks:
        raw_text = block["text"]
        metadata = block["metadata"]
        
        # Split text into small digestible chunks
        text_chunks = split_text(raw_text)
        
        for chunk in text_chunks:
            final_texts.append(chunk)
            final_metadatas.append(metadata.copy())
            
    # 3. Add to Database
    if final_texts:
        vector_store.add_documents(final_texts, final_metadatas)
        
    return len(final_texts)


def ingest_url(url: str, vector_store: BaseVectorStore) -> int:
    """
    Scrapes a web address, chunks its contents, generates embeddings,
    and indexes it in the vector database.
    Returns the total count of vector chunks indexed.
    """
    # 1. Scrape Web Page
    parsed_blocks = scrape_url(url)
    if not parsed_blocks:
        return 0
        
    final_texts = []
    final_metadatas = []
    
    # 2. Chunk Scraped Content
    for block in parsed_blocks:
        raw_text = block["text"]
        metadata = block["metadata"]
        
        text_chunks = split_text(raw_text)
        for chunk in text_chunks:
            final_texts.append(chunk)
            final_metadatas.append(metadata.copy())
            
    # 3. Add to Database
    if final_texts:
        vector_store.add_documents(final_texts, final_metadatas)
        
    return len(final_texts)
