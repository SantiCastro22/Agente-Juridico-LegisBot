import streamlit as st
import os
from main import agent, guardar_documento_generado

def main():
    st.title("Agente Jurídico Inteligente")
    consulta = st.text_area("Ingrese su consulta:")
    consultar = st.button("Consultar")
    
    if consultar:
        if not consulta.strip():
            st.warning("Ingrese una consulta para continuar.")
            return
            
        try:
            with st.spinner("Procesando su consulta..."):
                respuesta = agent.invoke(consulta)
                respuesta_str = respuesta["output"] if isinstance(respuesta, dict) and "output" in respuesta else str(respuesta)
                
                if respuesta_str.startswith("Error:"):
                    st.error(respuesta_str)
                    return
                    
                st.subheader("Respuesta:")
                st.write(respuesta_str)
                
                if respuesta_str.startswith("[Plantilla seleccionada: "):
                    try:
                        output_path = guardar_documento_generado(respuesta_str)
                        if output_path and os.path.exists(output_path):
                            st.success(f"Documento generado y guardado exitosamente en: {output_path}")
                            
                            # Mostrar el contenido del archivo guardado
                            with open(output_path, "r", encoding="utf-8") as f:
                                contenido = f.read()
                                st.download_button(
                                    "Descargar documento",
                                    contenido,
                                    file_name=os.path.basename(output_path),
                                    mime="text/plain"
                                )
                        else:
                            st.error("El documento se generó pero no se pudo guardar correctamente.")
                    except Exception as e:
                        st.error(f"Error al guardar el documento: {str(e)}")
        except Exception as e:
            st.error(f"Error al procesar la consulta: {str(e)}")

if __name__ == "__main__":
    main()
