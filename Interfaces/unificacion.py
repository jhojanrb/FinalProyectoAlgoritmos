import streamlit as st
import os
import sys

# Importar funciones necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Scrape.Unificar import read_bibtex, unify_results_from_files, save_bibtex, save_duplicates

def unification_view():
    """Vista para la unificación de archivos BibTeX."""
    st.title("Unificación y Detección de Duplicados en Archivos BibTeX")
    st.subheader("Sube los archivos generados durante el web scraping para unificarlos y detectar duplicados.")
    st.write("Asegúrate de que los archivos estén en formato BibTeX.")

    # Subida de archivos
    uploaded_files = st.file_uploader(
        "Sube los archivos BibTeX que se generaron durante el web scraping", type=["bib"], accept_multiple_files=True
    )

    st.subheader("Si no tienes los archivos BibTeX del Web Scraping, los puedes descargar a continuación")
    
    # Archivos BibTeX predefinidos
    bibtex_files = {
        "resultados_ACM.bib": "Contenido de ejemplo para ACM BibTeX.",
        "resultados_ieee.bib": "Contenido de ejemplo para IEEE BibTeX.",
        "resultados_springer_open.bib": "Contenido de ejemplo para Springer Open BibTeX.",
        "resultados_Sage.bib": "Contenido de ejemplo para Sage BibTeX."
    }

    for filename, content in bibtex_files.items():
        st.download_button(
            label=f"Descargar {filename}",
            data=content,
            file_name=filename,
            mime="text/plain"
        )

    st.write("Asegurate de subir exactamente los 4 archivos para asi generar los duplicados y unificados")
    if uploaded_files and len(uploaded_files) == 4:
        st.success("Archivos cargados correctamente. Procesando...")

        # Crear directorio temporal para los archivos
        temp_dir = "temp_bib"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, mode="w", encoding="utf-8") as f:
                f.write(uploaded_file.getvalue().decode("utf-8"))
            file_paths.append(file_path)

        # Procesar unificación
        unify_results_from_files(*file_paths)

        # Resultados
        st.success("¡Procesamiento completo!")
        st.write("Archivos generados:")
        st.download_button(
            "Descargar Unificados", 
            data=open("Data/unificados.bib", "r", encoding="utf-8").read(), 
            file_name="unificados.bib"
        )
        st.download_button(
            "Descargar Duplicados", 
            data=open("Data/duplicados.bib", "r", encoding="utf-8").read(), 
            file_name="duplicados.bib"
        )

        # Limpiar archivos temporales
        for file_path in file_paths:
            os.remove(file_path)
    else:
        st.info("Por favor, sube exactamente 4 archivos BibTeX para procesarlos.")


    # Botón para volver al home
    if st.button("Volver al Home"):
        st.session_state.current_view = "home"
