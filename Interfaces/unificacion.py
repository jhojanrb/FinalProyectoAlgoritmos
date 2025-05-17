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
    
    # Ruta a los archivos BibTeX en tu sistema
    base_path = "C:/2025-1/Analisis AlgoritmosProyectoFinal/FinalProyectoAlgoritmos/Data/resultados_ACM.bib"
    bibtex_files = {
        "resultados_ACM.bib": os.path.join(base_path, "resultados_ACM.bib"),
        "resultados_ieee.bib": os.path.join(base_path, "resultados_ieee.bib"),
        "resultados_springer_open.bib": os.path.join(base_path, "resultados_springer_open.bib"),
        "resultados_Sage.bib": os.path.join(base_path, "resultados_Sage.bib")
    }

    for filename, filepath in bibtex_files.items():
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                st.download_button(
                    label=f"Descargar {filename}",
                    data=file.read(),
                    file_name=filename,
                    mime="text/plain"
                )
        else:
            st.warning(f"El archivo {filename} no existe en la ruta especificada.")

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
