import requests
import json
import base64
import os
from typing import List
from dotenv import load_dotenv
import PyPDF2
import docx


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "models/gemini-1.5-flash-001"
CACHE_ID_FILE = "cache/gemini_cache_id.txt"

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_path: str) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_all_texts_from_data() -> str:
    data_dir = "docs"
    all_texts: List[str] = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.lower().endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                all_texts.append(f.read())
        elif fname.lower().endswith(".pdf"):
            all_texts.append(extract_text_from_pdf(fpath))
        elif fname.lower().endswith(".docx"):
            all_texts.append(extract_text_from_docx(fpath))
    return "\n\n".join(all_texts)

def is_cache_valid(cache_id: str) -> bool:
    # Gemini no da endpoint directo para validar, pero podemos intentar una consulta dummy
    query_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key={API_KEY}"
    query_body = {
        "contents": [
            {"role": "user", "parts": [{"text": "ping"}]}
        ],
        "cachedContent": cache_id
    }
    resp = requests.post(query_url, json=query_body)
    if resp.status_code == 200:
        return True
    # Si el error es por expiración o cache inválido, retorna False
    try:
        err = resp.json().get("error", {})
        if err.get("status", "").upper() in ["NOT_FOUND", "FAILED_PRECONDITION", "INVALID_ARGUMENT"]:
            return False
    except Exception:
        pass
    return False

# 1. Leer todos los documentos y codificarlos en base64
all_text = load_all_texts_from_data()
documento_bytes = all_text.encode("utf-8")
documento_b64 = base64.b64encode(documento_bytes).decode("utf-8")

# 2. Reutilizar cache si existe y es válido
if os.path.exists(CACHE_ID_FILE):
    with open(CACHE_ID_FILE) as f:
        cache_id = f.read().strip()
    if is_cache_valid(cache_id):
        print(f"Reutilizando cache existente: {cache_id}")
    else:
        print("Cache expirado o inválido, creando uno nuevo...")
        cache_id = None
else:
    cache_id = None

if not cache_id:
    cache_url = f"https://generativelanguage.googleapis.com/v1beta/cachedContents?key={API_KEY}"
    cache_body = {
        "model": MODEL,
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "text/plain",
                            "data": documento_b64
                        }
                    }
                ],
                "role": "user"
            }
        ],
        "systemInstruction": {
            "parts": [{"text": "You are an expert analyzing transcripts."}]
        },
        "ttl": "3600s"
    }
    cache_resp = requests.post(cache_url, json=cache_body)
    print("Respuesta completa:", cache_resp.text)
    cache_resp.raise_for_status()
    cache_id = cache_resp.json()["name"]
    print("Cache creado:", cache_id)
    os.makedirs(os.path.dirname(CACHE_ID_FILE), exist_ok=True)
    with open(CACHE_ID_FILE, "w") as f:
        f.write(cache_id)

# 3. Hacer consultas interactivas usando el cache
while True:
    pregunta = input("Pregunta (o 'salir'): ")
    if pregunta.strip().lower() in ["salir", "exit", "quit"]:
        break
    query_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key={API_KEY}"
    query_body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": pregunta}]
            }
        ],
        "cachedContent": cache_id
    }
    query_resp = requests.post(query_url, json=query_body)
    try:
        query_resp.raise_for_status()
        respuesta = query_resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("Respuesta:", respuesta)
    except Exception as e:
        print("Error en la consulta:", query_resp.text)