import os
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

class CAGModule:
    def __init__(self, openai_api_base: str, openai_api_key: str, model_name: str):
        self.model_name = model_name
        self.llm = OpenAI(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.0,
            max_tokens=300
        )

    def prepare_kvcache(self, documents: str|list, kvcache_path: str = None, answer_instruction: str = None):
        # Optimización: unir documentos y limpiar espacios
        if answer_instruction is None:
            answer_instruction = "Responder la pregunta de forma concisa y precisa."
        if isinstance(documents, list):
            documents = '\n'.join([d.strip() for d in documents if d.strip()])
        elif isinstance(documents, str):
            documents = documents.strip()
        else:
            raise ValueError("El parámetro documentos debe ser una cadena o una lista de cadenas.")
        knowledges = f"""
Dar respuestas precisas según el contexto dado.\n
La información del contexto se encuentra a continuación.\n------------------------------------------------\n{documents}\n------------------------------------------------\n{answer_instruction}\nPregunta:"""
        # Simulación de cache: guardar el prompt base
        if kvcache_path:
            with open(kvcache_path, 'w', encoding='utf-8') as f:
                f.write(knowledges)
        return knowledges

    def run_qna(self, question, knowledge_cache):
        # Optimización: prompt más robusto y limpio
        prompt = f"{knowledge_cache}\n{question}\nRespuesta:"
        response = self.llm.invoke(prompt)
        return response

def load_documents_with_langchain(path: str) -> list:
    import os
    docs = []
    if os.path.isdir(path):
        loaders = [
            (".txt", TextLoader),
            (".pdf", PyPDFLoader),
            (".docx", Docx2txtLoader)
        ]
        for ext, Loader in loaders:
            loader = DirectoryLoader(path, glob=f"*{ext}", loader_cls=Loader)
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"[ADVERTENCIA] No se pudo cargar {ext}: {e}")
    else:
        # Detecta extensión y usa el loader adecuado
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
            elif ext == ".pdf":
                loader = PyPDFLoader(path)
                docs = loader.load()
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
                docs = loader.load()
            else:
                raise ValueError(f"Extensión de archivo no soportada: {ext}")
        except Exception as e:
            print(f"ERROR No se pudo cargar el archivo {path}: {e}")
            return []
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(docs)
