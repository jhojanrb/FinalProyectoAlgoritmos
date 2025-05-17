import streamlit as st
import pandas as pd
import os
from PIL import Image
import tempfile
import shutil

# Importar funciones necesarias
from Requerimiento2.graficos import generate_and_save_charts
from Requerimiento2.limpieza_normalizacion import normalize_authors, clean_journal_name, normalize_product_type, parse_large_bib
from Requerimiento2.generar_estadisticas import generate_statistics, save_statistics

def estadisticas_view():
    st.title("Generación de Estadísticas y Gráficos")
    st.subheader("En esta sección puedes subir un archivo BibTeX y generar estadísticas y gráficos a partir de él.")
    st.write("Las estadísticas generadas incluyen:")
    st.write("- Total de publicaciones")
    st.write("- Distribución por tipo de publicación")
    st.write("- Evolución de publicaciones por tipo")
    st.write("- 15 autor(es) más frecuentes")
    st.write("- 15 journal(s) más frecuentes")
    st.write("- 15 publisher(s) más frecuentes")
 

    # Crear directorio temporal con tempfile (manejo más seguro)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Selección del archivo
        uploaded_file = st.file_uploader("Sube el archivo BibTeX unificados", type=["bib"])
        if uploaded_file:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            # Guardar el archivo subido
            with open(file_path, mode="w", encoding="utf-8") as f:
                f.write(uploaded_file.getvalue().decode("utf-8"))

            # Paso 1: Parsear el archivo BibTeX y crear el DataFrame
            entries = parse_large_bib(file_path)
            df = pd.DataFrame(entries)

            # Paso 2: Normalización de datos
            with st.spinner("Normalizando datos..."):
                df['author'] = df['author'].apply(normalize_authors)
                df['tipo_normalizado'] = df['tipo'].apply(normalize_product_type)

                if 'journal' in df.columns:
                    df['journal'] = df['journal'].apply(clean_journal_name)
                if 'publisher' in df.columns:
                    df['publisher'] = df['publisher'].apply(clean_journal_name)

                # Limpieza de años
                if 'year' in df.columns:
                    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')[0]
                    valid_years = df['year'].notna()
                    st.write(f"- Publicaciones con año válido: {valid_years.sum()}/{len(df)}")

            # Paso 3: Generar estadísticas
            stats = generate_statistics(df)

            # Mostrar estadísticas básicas
            st.subheader("Resumen de Estadísticas")
            st.write(f"- Total publicaciones: {len(df)}")
            st.write("- Distribución por tipo normalizado:")
            st.dataframe(df['tipo_normalizado'].value_counts())

            # Paso 4: Exportar resultados
            output_stats_path = os.path.join(temp_dir, "estadisticas_finales.xlsx")
            folder_graficos = temp_dir

            # Generar y guardar gráficos como imágenes
            generate_and_save_charts(stats, folder_graficos)
            save_statistics(stats, output_stats_path)

            # Mostrar gráficos
            st.subheader("Gráficos Generados")
            for chart_file in os.listdir(folder_graficos):
                if chart_file.endswith(".png"):
                    image_path = os.path.join(folder_graficos, chart_file)
                    try:
                        image = Image.open(image_path)
                        st.image(image, caption=chart_file, use_container_width=True)
                        image.close()  # Cerrar el archivo de imagen
                    except Exception as e:
                        st.warning(f"No se pudo mostrar {chart_file}: {str(e)}")

            # Mostrar y descargar el archivo Excel
            st.subheader("Archivo de Estadísticas")
            st.write("Selecciona la hoja para visualizar:")

            # Leer el archivo Excel y obtener los nombres de las hojas
            excel_data = pd.ExcelFile(output_stats_path)
            sheet_name = st.selectbox("Hojas disponibles", excel_data.sheet_names)

            # Mostrar la hoja seleccionada
            if sheet_name:
                df_sheet = excel_data.parse(sheet_name)
                st.dataframe(df_sheet)

            # Botón de descarga - Leer el archivo en modo binario
            with open(output_stats_path, "rb") as f:
                excel_data_bytes = f.read()
            
            st.download_button(
                "Descargar Estadísticas en Excel",
                data=excel_data_bytes,
                file_name="estadisticas_finales.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    finally:
        # Limpieza segura del directorio temporal
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            st.warning(f"No se pudo limpiar completamente el directorio temporal: {str(e)}")

    # Botón para volver al home
    if st.button("Volver al Home"):
        st.session_state.current_view = "home"