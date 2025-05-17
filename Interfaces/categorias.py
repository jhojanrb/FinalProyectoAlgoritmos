import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os

from Requerimiento3.categorias import keywords
from Requerimiento3.graficos import plot_bar_chart, generate_wordcloud, plot_cooccurrence_network, cargarPalabras_excel  
from Requerimiento3.normalizacion_lectura import load_bibtex, count_keywords

# Función para guardar palabras clave en un archivo Excel
def guardar_keywords_en_excel(keyword_data, output_path):
    df = pd.DataFrame(keyword_data)
    df = df.sort_values(by=["Categoría", "Frecuencia"], ascending=[True, False])
    df.to_excel(output_path, index=False)

def categorias_view():
    st.title("Análisis de Palabras Clave Categorizadas")
    st.subheader("Análisis de Palabras Clave en Abstracts, graficos de wordcloud y co-ocurrencia")
    st.write("Con base en ciertas categorias y variables, se presentaran los resultados de la frecuencia de estas mismas en los abstracts de los documentos subidos.")
    st.write("Se generará un archivo Excel con las frecuencias de palabras clave y se mostrarán gráficos de barras, nube de palabras y red de co-ocurrencia.")

    
    # Cargar archivo BibTeX
    uploaded_file = st.file_uploader("Sube el archivo BibTeX unificados", type=["bib"])
    
    if uploaded_file:
        # Crear carpeta temporal para gráficos y archivos
        temp_dir = "temp_categorias"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # Guardar archivo subido
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(uploaded_file.getvalue().decode("utf-8"))
        
        try:
            # Cargar abstracts
            abstracts = load_bibtex(file_path)
            
            if not abstracts:
                st.warning("No se encontraron abstracts en el archivo. Verifica que los campos 'abstract' existan.")
                return

            # Contar palabras clave
            keyword_data, keyword_counts = count_keywords(abstracts, keywords)
            
            if not keyword_counts:
                st.warning("No se encontraron coincidencias con las palabras clave. Verifica los abstracts.")
                return
            
            # Mostrar frecuencias
            st.subheader("Frecuencias de Palabras Clave")
            st.dataframe(pd.DataFrame.from_dict(keyword_counts, orient="index", columns=["Frecuencia"]).reset_index().rename(columns={"index": "Palabra Clave"}))
            
            # Guardar resultados en Excel
            # Este excel se debe guardar en la carpeta temporal para asi generar el co-occurence network
            output_excel = os.path.join(temp_dir, "frecuencia_keywords_categorizadas.xlsx")
            guardar_keywords_en_excel(keyword_data, output_excel)
            st.write("Archivo Excel generado con las frecuencias de palabras clave para la realización de gráficos.")
            st.success(f"Archivo Excel guardado en: {output_excel}")
            

            # Mostrar y descargar el archivo Excel
            st.subheader("Archivo Excel de Frecuencias")
            with pd.ExcelFile(output_excel) as excel_data:
                st.dataframe(pd.read_excel(excel_data, sheet_name=0))
            
            # Botón para descargar el archivo Excel
            st.download_button(
                label="Descargar Excel de Frecuencias",
                data=open(output_excel, "rb").read(),
                file_name="frecuencia_keywords_categorizadas.xlsx"
            )
            
            # Graficar resultados
            st.header("Gráficos Generados")
            
            # Gráfico de barras
            st.subheader("Gráfico de Barras")
            st.write("Este grafico muestra el top 20 de palabras clave y su frecuencia.")
            temp_file = plot_bar_chart(keyword_counts, temp_dir)
            st.image(temp_file, caption="Top 20 - Frecuencia de Palabras Clave", use_container_width=True)

            # Nube de palabras
            st.subheader("Nube de Palabras")
            temp_file_wc = generate_wordcloud(keyword_counts, temp_dir)
            st.image(temp_file_wc, caption="Nube de Palabras", use_container_width=True)

            # Red de co-ocurrencia
            st.subheader("Red de Co-ocurrencia")
            st.write("Este grafico muestra la co-ocurrencia de palabras clave en los abstracts.")
            keywords_by_category = cargarPalabras_excel(output_excel)
            temp_file_net = plot_cooccurrence_network(keywords_by_category, temp_dir)
            st.image(temp_file_net, caption="Red de Co-ocurrencia", use_container_width=True)
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
        
        finally:
            # Limpiar archivos temporales
            try:
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
            except Exception as e:
                st.error(f"No se pudieron limpiar los archivos temporales: {str(e)}")

    # Boton para regresar a la vista de inicio
    if st.button("Volver al Home"):
        st.session_state.current_view = "home"
