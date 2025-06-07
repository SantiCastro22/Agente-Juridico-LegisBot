import os
import requests
import json
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from helpers import get_env_var
from langchain.schema import Document

# Wrapper manual para embeddings LM Studio
class LMStudioEmbeddings:
    def __init__(self, api_base, model_name):
        self.url = f"{api_base}/embeddings"
        self.model = model_name

    def embed_documents(self, texts):
        payload = {
            "model": self.model,
            "input": texts
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, data=json.dumps(payload), headers=headers)
        if response.status_code != 200:
            raise Exception(f"Embeddings error: {response.status_code} - {response.text}")
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def split_by_articulos(text):
    # Divide el texto por artículos (Art. 1, ARTÍCULO 1, etc. variantes)
    pattern = r"(?im)(art[íi]culo\s*\d+[\wº°]*|art\.\s*\d+[\wº°]*)"
    matches = list(re.finditer(pattern, text))
    articulos = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        articulo_text = text[start:end].strip()
        if articulo_text:
            articulos.append(articulo_text)
    # DEBUG: Mostrar los primeros artículos extraídos
    print("[DEBUG] Primeros artículos extraídos:")
    for idx, art in enumerate(articulos[:10]):
        print(f"[ART {idx+1}] {art[:120].replace('\n',' ')} ...")
    print(f"[DEBUG] Total artículos extraídos: {len(articulos)}")
    return articulos if articulos else [text]


def load_all_documents(data_dir: str):
    loaders = [
        (".txt", TextLoader),
        (".pdf", PyPDFLoader),
        (".docx", Docx2txtLoader)
    ]
    docs = []
    for ext, Loader in loaders:
        loader = DirectoryLoader(data_dir, glob=f"*{ext}", loader_cls=Loader)
        loaded = loader.load()
        # Si es legislación, dividir por artículos
        for d in loaded:
            content = getattr(d, 'page_content', None)
            meta = getattr(d, 'metadata', {})
            if content and isinstance(content, str):
                # Si el nombre del archivo sugiere código/ley, dividir por artículos
                filename = meta.get('source', '').lower()
                if any(x in filename for x in ["codigo", "código", "ley", "constitucion", "cpc"]):
                    articulos = split_by_articulos(content)
                    for art in articulos:
                        docs.append(Document(page_content=art, metadata=meta))
                else:
                    docs.append(d)
    return docs


def build_rag_chain(data_dir: str, persist_path: str = None):
    # Cargar documentos
    docs = load_all_documents(data_dir)
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    # Filtrar splits para asegurar que el contenido es string y no None
    valid_texts = []
    valid_metadatas = []
    for d in splits:
        content = getattr(d, 'page_content', None)
        meta = getattr(d, 'metadata', {})
        if isinstance(content, str):
            content = content.strip()
            if content:
                valid_texts.append(content)
                valid_metadatas.append(meta if isinstance(meta, dict) else {})
    # DEBUG: Mostrar tipos y longitud
    print(f"[DEBUG] valid_texts: {len(valid_texts)} elementos, tipos: {[type(t) for t in valid_texts][:5]}")
    print(f"[DEBUG] valid_metadatas: {len(valid_metadatas)} elementos")
    # Asegurar que ambas listas tengan la misma longitud
    if len(valid_texts) != len(valid_metadatas):
        min_len = min(len(valid_texts), len(valid_metadatas))
        valid_texts = valid_texts[:min_len]
        valid_metadatas = valid_metadatas[:min_len]
    # Embeddings y vectorstore
    embeddings = LMStudioEmbeddings(
        api_base=get_env_var("OPENAI_API_BASE"),
        model_name=get_env_var("EMBEDDING_MODEL_NAME")
    )
    persist_dir = persist_path or "chroma_db"
    os.makedirs(persist_dir, exist_ok=True)
    # Solo pasar strings planos
    valid_texts = [t for t in valid_texts if isinstance(t, str)]
    # Recortar textos a 8191 caracteres (límite típico de modelos de embeddings)
    texts_for_embeddings = [t[:8191] for t in valid_texts if isinstance(t, str) and t.strip()]
    print(f"DEBUG: tipos y longitud de textos para embeddings: {[type(t) for t in texts_for_embeddings]}, {len(texts_for_embeddings)}")
    print("DEBUG: texts_for_embeddings =", texts_for_embeddings)
    print("DEBUG: type(texts_for_embeddings) =", type(texts_for_embeddings))
    if texts_for_embeddings:
        print("DEBUG: type(texts_for_embeddings[0]) =", type(texts_for_embeddings[0]))
    for idx, t in enumerate(texts_for_embeddings):
        print(f"[DEBUG] Texto {idx}: longitud={len(t)}, inicio='{t[:60]}', fin='{t[-60:]}'")
    # Antes de pasar a embeddings:
    texts_for_embeddings = [str(t) for t in texts_for_embeddings if t and isinstance(t, str)]
    print("DEBUG: texts_for_embeddings (final) =", texts_for_embeddings)
    # Obtener embeddings manualmente
    all_embeddings = embeddings.embed_documents(texts_for_embeddings)
    # Crear Chroma usando la función de embeddings personalizada
    docs_for_chroma = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts_for_embeddings, valid_metadatas)
    ]
    vectordb = Chroma.from_documents(
        docs_for_chroma,
        embeddings,
        persist_directory=persist_dir
    )
    # LLM
    llm = OpenAI(
        openai_api_key=get_env_var("OPENAI_API_KEY"),
        openai_api_base=get_env_var("OPENAI_API_BASE"),
        model_name=get_env_var("MODEL_NAME")
    )
    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    return qa_chain

if __name__ == "__main__":
    data_dir = os.path.join("docs", "clientes")
    print("Construyendo pipeline RAG sobre:", data_dir)
    rag_chain = build_rag_chain(data_dir)
    print("Agente RAG listo. Escribe tu pregunta:")
    while True:
        q = input("Pregunta: ")
        if q.lower() in ["salir", "exit", "quit"]:
            break
        a = rag_chain.invoke({"query": q})
        print("Respuesta:", a["result"])


