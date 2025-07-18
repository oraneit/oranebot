import os
import pickle
import csv
import numpy as np
from datetime import datetime
import secrets
import logging
import subprocess
import sys
from collections import defaultdict

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ─── Setup Logging ───
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Load Environment Variables ───
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "changeme")
CREDS_PATH = os.getenv("GSPREAD_CREDENTIALS")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")

# ─── Initialize OpenAI Client ───
if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY environment variable not set.")
    sys.exit(1)
client = OpenAI(api_key=OPENAI_API_KEY)

# ─── FastAPI Setup ───
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.prodifyteam.com", "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── FAISS Index & System Prompt ───
INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4-turbo"
MAX_PROMPTS_PER_SESSION = 10

session_prompt_counts = defaultdict(int)

def load_system_prompt():
    """Loads the system prompt and appends the prompt bank."""
    try:
        with open("prompts/system_prompt.txt", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        logger.error("CRITICAL: prompts/system_prompt.txt not found.")
        return "You are a helpful assistant."  # Fallback prompt

    prompt_bank = []
    csv_path = os.path.join("data", "prompt_bank.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tmpl = row.get("template", "").strip()
                    if tmpl:
                        prompt_bank.append(tmpl)
        except Exception as e:
            logger.error(f"Failed to load or parse prompt_bank.csv: {e}")

    if prompt_bank:
        prompt_bank_text = "\n".join(prompt_bank)
        system_prompt += f"\n\n### Prompt Bank Templates\n{prompt_bank_text}"
    
    return system_prompt

SYSTEM_PROMPT = load_system_prompt()

# Load FAISS index + chunks at startup
try:
    import faiss
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Successfully loaded FAISS index with {len(chunks)} chunks.")
    else:
        index, chunks = None, []
        logger.warning("FAISS index or chunks file not found. Context-based responses will be disabled.")
except (ModuleNotFoundError, ImportError):
    index, chunks = None, []
    logger.warning("FAISS library not found. Context-based responses are disabled.")
except Exception as e:
    index, chunks = None, []
    logger.error(f"An error occurred while loading the FAISS index: {e}")

def get_context(query: str, k: int = 3) -> str:
    """Retrieves relevant text chunks from the FAISS index based on the user's query."""
    if not index or not chunks:
        return ""
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
        vector = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
        _, I = index.search(vector, k)
        return "\n\n".join(chunks[i] for i in I[0])
    except Exception as e:
        logger.error(f"Error during FAISS context retrieval: {e}")
        return ""

# ─── Google Sheets Setup ───
try:
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_PATH, scope)
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SPREADSHEET_ID)
except Exception as e:
    logger.critical(f"Could not connect to Google Sheets. Check credentials and SPREADSHEET_ID. Error: {e}")
    sheet = None

def get_worksheet(name: str):
    """Gets a worksheet by name, creating it with headers if it doesn't exist."""
    if not sheet:
        raise HTTPException(status_code=500, detail="Google Sheets service is not available.")
    try:
        return sheet.worksheet(name)
    except gspread.WorksheetNotFound:
        try:
            ws = sheet.add_worksheet(title=name, rows="1000", cols="20")
            headers = ["Session ID", "Timestamp", "Role", "Message", "Intent", "Quality", "Notes"]
            if name == "Leads":
                headers = ["Session ID", "Timestamp", "Name", "Email", "Details", "Conversation"]
            ws.append_row(headers)
            return ws
        except gspread.exceptions.APIError as e:
            logger.error(f"Google Sheets API error when creating worksheet '{name}': {e}")
            raise HTTPException(status_code=500, detail=f"Could not create worksheet '{name}'.")
    except gspread.exceptions.APIError as e:
        logger.error(f"Google Sheets API error when getting worksheet '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not access worksheet '{name}'.")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Handles the main chat interaction, including prompt limiting and context retrieval."""
    try:
        data = await request.json()
        user_msg = data.get("message", "")
        session_id = data.get("session_id", "unknown")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body.")

    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if session_id == "unknown":
        raise HTTPException(status_code=400, detail="Session ID is required.")

    if session_prompt_counts[session_id] >= MAX_PROMPTS_PER_SESSION:
        raise HTTPException(status_code=429, detail="You have reached the message limit for this session.")

    session_prompt_counts[session_id] += 1
    ts = datetime.utcnow().isoformat()

    try:
        ws_chat = get_worksheet("Chats")
        ws_chat.append_row([session_id, ts, "user", user_msg, "", "", ""])
    except HTTPException as e:
        logger.error(f"Failed to log user message to Google Sheets for session {session_id}: {e.detail}")

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = get_context(user_msg)
    if ctx:
        msgs.append({"role": "system", "content": f"Context from knowledge base:\n{ctx}"})
    msgs.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=msgs,
            max_tokens=250,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=503, detail="The AI service is currently unavailable.")

    try:
        ws_chat.append_row([session_id, ts, "assistant", answer, "", "", ""])
    except HTTPException as e:
        logger.error(f"Failed to log assistant response to Google Sheets for session {session_id}: {e.detail}")

    return {"reply": answer}

@app.post("/lead")
async def receive_lead(request: Request):
    """Receives lead data and appends it to the 'Leads' Google Sheet."""
    try:
        data = await request.json()
        session_id = data.get("session_id", "unknown")
        name = data.get("name", "")
        email = data.get("email", "")
        details = data.get("details", "")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body.")

    if not all([session_id != "unknown", name, email, details]):
        raise HTTPException(status_code=400, detail="Missing required lead information.")

    ts = datetime.utcnow().isoformat()
    conversation = ""
    try:
        ws_chat = get_worksheet("Chats")
        cells = ws_chat.findall(session_id, in_column=1)
        conv_rows = []
        for cell in cells:
            row_values = ws_chat.row_values(cell.row)
            if len(row_values) >= 4:
                conv_rows.append(f"{row_values[2]}: {row_values[3]}")
        conversation = "\n".join(conv_rows)
    except Exception as e:
        logger.error(f"Could not retrieve conversation history for lead {session_id}: {e}")
        conversation = "Error retrieving conversation history."

    try:
        ws_lead = get_worksheet("Leads")
        ws_lead.append_row([session_id, ts, name, email, details, conversation])
    except HTTPException as e:
        logger.error(f"Failed to write lead to Google Sheets for session {session_id}: {e.detail}")
        raise HTTPException(status_code=500, detail="Could not save lead information.")

    return {"status": "received"}

@app.post("/admin/reindex")
async def admin_reindex(x_admin_key: str = Header(None)):
    """Triggers the data ingestion script. NOTE: The server must be restarted to load the new index."""
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing admin key.")
    
    logger.info("Admin request received to re-index data.")
    
    try:
        backend_dir = os.path.dirname(__file__)
        ingest_script_path = os.path.join(backend_dir, "ingest.py")
        
        if not os.path.exists(ingest_script_path):
            logger.error(f"ingest.py not found at {ingest_script_path}")
            raise HTTPException(status_code=500, detail="Ingestion script not found.")

        result = subprocess.run(
            [sys.executable, ingest_script_path],
            cwd=backend_dir,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Ingestion script stdout:\n{result.stdout}")
        logger.info(f"Ingestion script stderr:\n{result.stderr}")
        
        return {
            "status": "Ingestion script completed successfully.",
            "message": "IMPORTANT: Please restart the server application to load the newly created index.",
            "output": result.stdout
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Ingestion script failed with exit code {e.returncode}. Stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed. Check server logs. Stderr: {e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during re-indexing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during re-indexing: {e}")