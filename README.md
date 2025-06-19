# Agente Jurídico - LegisBot

Agente inteligente para estudios jurídicos que combina RAG (búsqueda aumentada por recuperación) y CAG (generación de documentos personalizados a partir de plantillas y datos de clientes), usando modelos LLM locales (OpenAI compatible, JAM, LM Studio) o en la nube (Gemini), con integración LangChain y Streamlit.

---

## Estructura del Proyecto

```
├── app.py                  # Interfaz Streamlit (opcional)
├── main.py                 # Lógica principal del agente y CLI
├── cag.py                  # Módulo CAG (generación de documentos)
├── rag.py                  # Módulo RAG (búsqueda en clientes y legislación)
├── utils.py                # Utilidades: extracción de datos, reemplazo de placeholders
├── helpers.py              # Utilidades para variables de entorno
├── requirements.txt        # Dependencias
├── chroma_db_clientes/     # Vectorstore persistente de clientes
├── chroma_db_legislacion/  # Vectorstore persistente de legislación
├── docs/
│   ├── clientes/           # Documentos y datos de clientes (ej: Datos del Cliente.docx)
│   ├── legislacionLR/      # Legislación, códigos, constitución
│   ├── plantillas/         # Plantillas y contratos jurídicos
│   └── ...                 # Otros documentos
├── docs_outputs/           # Documentos generados
```

---

## Instalación y Configuración

1. **Clona el repositorio y entra al directorio:**
   ```sh
   git clone <repo-url>
   cd agente juridico
   ```

2. **Crea un entorno virtual y activa:**
   ```sh
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Instala las dependencias:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Configura las variables de entorno en un archivo `.env`:**
   Ver ejemplo .env-example:
   

5. **Coloca tus documentos:**
   - Clientes: en `docs/clientes/`
   - Legislación: en `docs/legislacionLR/`
   - Plantillas: en `docs/plantillas/`

---

## Modelos soportados

- **Locales:** JAM, LM Studio, Ollama, u otros compatibles con la API OpenAI (`/v1/completions`, `/v1/chat/completions`).

---

