from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAI
from rag import build_rag_chain
from cag import CAGModule, load_documents_with_langchain
from helpers import get_env_var
from utils import extraer_datos_cliente, reemplazar_placeholders
import os
from docx import Document
import re

# Configuración LLM
llm = OpenAI(
    openai_api_base=get_env_var("OPENAI_API_BASE"),
    openai_api_key=get_env_var("OPENAI_API_KEY"),
    model_name=get_env_var("MODEL_NAME"),
    temperature=0.0,
    max_tokens=300
)

# Herramienta RAG: búsqueda en clientes
def rag_clientes_tool_func(query):
    if not hasattr(rag_clientes_tool_func, "chain"):
        rag_clientes_tool_func.chain = build_rag_chain(os.path.join("docs", "clientes"), persist_path="chroma_db_clientes")
    return rag_clientes_tool_func.chain.invoke({"query": query})["result"]

# Herramienta RAG: búsqueda en legislación/plantillas
def rag_legislacion_tool_func(query):
    if not hasattr(rag_legislacion_tool_func, "chain"):
        rag_legislacion_tool_func.chain = build_rag_chain(os.path.join("docs", "legislacionLR"), persist_path="chroma_db_legislacion")
    return rag_legislacion_tool_func.chain.invoke({"query": query})["result"]

# Herramienta CAG: consulta/generación sobre plantillas
def cag_tool_func(query, plantilla_name=None):
    try:
        plantillas_dir = os.path.join("docs", "plantillas")
        plantillas = [f for f in os.listdir(plantillas_dir) if f.lower().endswith((".txt", ".docx", ".pdf"))]
        
        # Selección de plantilla según la consulta
        if plantilla_name is None:
            query_lower = query.lower()
            # Selección directa si el prompt contiene "prescripción"
            if "prescripción" in query_lower or "prescripcion" in query_lower:
                plantilla_name = "PLANTILLA PROMUEVE DEMANDA DE PRESCRIPCIÓN.docx"
            else:
                import difflib
                match = difflib.get_close_matches(query_lower, [p.lower() for p in plantillas], n=1, cutoff=0.3)
                if match:
                    plantilla_name = next((p for p in plantillas if p.lower() == match[0]), plantillas[0])
                else:
                    plantilla_name = plantillas[0]  # fallback: la primera

        if not plantilla_name:
            return "No se encontró una plantilla adecuada para su consulta."

        plantilla_path = os.path.join(plantillas_dir, plantilla_name)
        if not os.path.exists(plantilla_path):
            return f"Error: No se encontró la plantilla {plantilla_name}"

        docs = load_documents_with_langchain(plantilla_path)
        if not docs:
            return f"Error: No se pudo cargar el contenido de la plantilla {plantilla_name}"

        docs_text = [d.page_content for d in docs]
        
        # Extraer datos del cliente y fusionar con plantilla
        datos_cliente_path = os.path.join("docs", "clientes", "Datos del Cliente.docx")
        if not os.path.exists(datos_cliente_path):
            return "Error: No se encontró el archivo de datos del cliente"
            
        datos_cliente = extraer_datos_cliente(datos_cliente_path)
        docs_text = [reemplazar_placeholders(t, datos_cliente) for t in docs_text]
        texto_fusionado = "\n\n".join(docs_text)

        # Limitar a 3000 caracteres
        MAX_CHARS = 3000
        if len(texto_fusionado) > MAX_CHARS:
            texto_fusionado = texto_fusionado[:MAX_CHARS] + "\n...[TEXTO RECORTADO]..."

        cag = CAGModule(
            get_env_var("OPENAI_API_BASE"),
            get_env_var("OPENAI_API_KEY"),
            get_env_var("MODEL_NAME")
        )
        
        knowledge_cache = cag.prepare_kvcache(docs_text)
        prompt_contexto = "\n".join([f"{k}: {v}" for k, v in datos_cliente.items()])
        prompt_final = f"{query}\n\nDatos del cliente:\n{prompt_contexto}\n\n{texto_fusionado}"
        
        MAX_PROMPT = 3500
        if len(prompt_final) > MAX_PROMPT:
            prompt_final = prompt_final[:MAX_PROMPT] + "\n...[PROMPT RECORTADO]..."
            
        respuesta_llm = cag.run_qna(prompt_final, knowledge_cache)
        return f"[Plantilla seleccionada: {plantilla_name}]\n{texto_fusionado}\n\n---\n{respuesta_llm}"
    except Exception as e:
        return f"Error al procesar la plantilla: {str(e)}"

tools = [
    Tool(
        name="Buscar en clientes",
        func=rag_clientes_tool_func,
        description="Usa esto para responder preguntas sobre expedientes, datos o documentos de clientes."
    ),
    Tool(
        name="Buscar en legislación y códigos",
        func=rag_legislacion_tool_func,
        description="Usa esto para responder preguntas sobre leyes, constituciones, códigos, normativas, o documentos legales."
    ),
    Tool(
        name="Consultar/generar plantilla",
        func=lambda q: cag_tool_func(q),
        description="Usa esto para generar o consultar plantillas y contratos jurídicos."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "Eres un agente jurídico experto. "
            "Cuando uses una herramienta, nunca incluyas 'Final Answer' en la misma respuesta que una acción. "
            "Solo responde con 'Final Answer' cuando realmente hayas terminado y tengas la respuesta final. "
            "No repitas la pregunta del usuario. Sé claro, preciso y profesional. "
            "Si usas una herramienta, espera el resultado antes de dar la respuesta final. "
            "No inventes información si no está en los documentos o contexto proporcionado. "
            "Si no encuentras la información, responde claramente que no está disponible en la base de datos."
            "La respuesta final siempre en Español. "
        )
    }
)

def guardar_documento_generado(respuesta_str):
    """Guarda el documento generado en docs_outputs con nombre único y limpio el encabezado."""
    try:
        import re
        import os
        from datetime import datetime
        
        # Asegurar que el directorio existe
        output_dir = "docs_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraer nombre de la plantilla
        match = re.match(r"\[Plantilla seleccionada: (.+?)\]", respuesta_str)
        if match:
            plantilla_name = match.group(1)
            safe_name = plantilla_name.replace(" ", "_").replace(".", "_")
        else:
            safe_name = "plantilla"
            
        # Generar nombre único con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{safe_name}_{timestamp}.txt")
        
        # Quitar el encabezado de plantilla seleccionada y limpiar el texto
        texto = re.sub(r"^\[Plantilla seleccionada: .+?\]\n?", "", respuesta_str)
        texto = texto.strip()
        
        # Guardar el archivo
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(texto)
            
        print(f"Documento guardado exitosamente en: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error al guardar el documento: {str(e)}")
        raise

if __name__ == "__main__":
    print("Agente jurídico inteligente listo. Escribe tu consulta:")
    while True:
        pregunta = input("Consulta: ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        try:
            respuesta = agent.invoke(pregunta)
            respuesta_str = respuesta["output"] if isinstance(respuesta, dict) and "output" in respuesta else str(respuesta)
            print("Respuesta:", respuesta_str)
            if respuesta_str.startswith("[Plantilla seleccionada: "):
                guardar_documento_generado(respuesta_str)
        except Exception as e:
            print(f"[ERROR] Consulta fallida: {e}")

