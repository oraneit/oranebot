import os
import glob
import pickle
import numpy as np
import logging
import sys
from dotenv import load_dotenv

# ─── Setup Logging ───
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Load Environment & Initialize Clients ───
load_dotenv()

try:
    from openai import OpenAI
    import faiss
    import pandas as pd
    from PyPDF2 import PdfReader
    from docx import Document
except ImportError as e:
    logger.critical(f"A required library is not installed: {e}. Please run 'pip install -r requirements.txt'")
    sys.exit(1)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY environment variable not set.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Constants ───
DATA_DIR = "data"
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 400  # Reduced for more granular chunks
CHUNK_OVERLAP = 50 # Added overlap for better context continuity

# ─── Text Extractors ───

def extract_pdf(path):
    """Extracts text from a PDF file, handling potential errors."""
    try:
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF {os.path.basename(path)}: {e}")
        return ""

def extract_docx(path):
    """Extracts text from a DOCX file, handling potential errors."""
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as e:
        logger.error(f"Failed to read DOCX {os.path.basename(path)}: {e}")
        return ""

def extract_xlsx(path):
    """Extracts structured data from an XLSX file, converting rows to descriptive text."""
    try:
        sheets = pd.read_excel(path, sheet_name=None)
        texts = []
        for sheet_name, df in sheets.items():
            # Sanitize column names
            df.columns = [str(c).strip() for c in df.columns]
            # Convert each row to a descriptive string
            for i, row in df.iterrows():
                row_desc = f"In sheet '{sheet_name}', row {i+1} contains: "
                row_texts = [f'{col} is \"{row[col]}\"' for col in df.columns if pd.notna(row[col])]
                row_desc += ", ".join(row_texts)
                texts.append(row_desc)
        return "\n".join(texts)
    except Exception as e:
        logger.error(f"Failed to process XLSX {os.path.basename(path)}: {e}")
        return ""

def extract_csv(path):
    """Extracts structured data from a CSV file, similar to XLSX."""
    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        # Sanitize column names
        df.columns = [str(c).strip() for c in df.columns]
        texts = []
        for i, row in df.iterrows():
            row_desc = f"In file '{os.path.basename(path)}', row {i+1} contains: "
            row_texts = [f'{col} is \"{row[col]}\"' for col in df.columns if pd.notna(row[col])]
            row_desc += ", ".join(row_texts)
            texts.append(row_desc)
        return "\n".join(texts)
    except Exception as e:
        logger.error(f"Failed to process CSV {os.path.basename(path)}: {e}")
        return ""

def extract_txt(path):
    """Extracts text from a plain text file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read TXT {os.path.basename(path)}: {e}")
        return ""

# ─── Core Logic ───

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def load_documents():
    """Loads and extracts text from all supported documents in the DATA_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_chunks = []
    
    # Added txt and csv to the list of supported formats
    extractors = {
        "pdf": extract_pdf,
        "docx": extract_docx,
        "xlsx": extract_xlsx,
        "csv": extract_csv,
        "txt": extract_txt,
    }

    for ext, extractor_func in extractors.items():
        filepaths = glob.glob(f"{DATA_DIR}/**/*.{ext}", recursive=True)
        for filepath in filepaths:
            logger.info(f"Reading {os.path.basename(filepath)}...")
            text = extractor_func(filepath)
            if text:
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                logger.info(f"→ Extracted and split into {len(chunks)} chunks.")
            else:
                logger.warning(f"→ No text extracted from {os.path.basename(filepath)}. Skipping.")
    
    return all_chunks

def build_index(chunks):
    """Builds and saves the FAISS index from text chunks."""
    if not chunks:
        logger.warning("No text chunks to process. Aborting index build.")
        return

    logger.info(f"Embedding {len(chunks)} chunks for the FAISS index...")
    
    try:
        # Call the API once for the entire batch of chunks
        response = client.embeddings.create(model=EMBED_MODEL, input=chunks)
        # Extract the embedding vector from each data object in the response
        embeddings = [embedding_data.embedding for embedding_data in response.data]
        embeddings = np.array(embeddings, dtype="float32")
    except Exception as e:
        logger.critical(f"OpenAI embedding API call failed: {e}")
        logger.critical("Could not generate embeddings. Aborting index build.")
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    logger.info("Persisting index and chunks to disk...")
    try:
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)
        faiss.write_index(index, INDEX_FILE)
        logger.info(f"Successfully built and saved index with {index.ntotal} vectors.")
    except Exception as e:
        logger.error(f"Failed to save index or chunks to disk: {e}")

if __name__ == "__main__":
    logger.info("Starting document ingestion process...")
    document_chunks = load_documents()
    build_index(document_chunks)
    logger.info("Ingestion process finished.")