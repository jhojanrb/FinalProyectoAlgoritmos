import streamlit as st
import os
import sys

# Importar funciones necesarias
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Scrape.Unificar import read_bibtex, unify_results_from_files, save_bibtex, save_duplicates

def unification_view():
    """Vista para la unificación de archivos BibTeX."""
    st.title("Unificación y Detección de Duplicados en Archivos BibTeX")

    # Botón para volver al home
    if st.button("Volver al Home"):
        st.session_state.current_view = "home"

    # Subida de archivos
    uploaded_files = st.file_uploader(
        "Sube los 4 archivos BibTeX que se generaron durante el web_scrapping", type=["bib"], accept_multiple_files=True
    )

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
