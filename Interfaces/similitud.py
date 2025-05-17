import streamlit as st
import nltk
import time
import os
import sys
from io import StringIO
import contextlib
import numpy as np

# Agregar la carpeta raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from normalizacion5 import load_bibtex
from algoritmos import tfidf_similarity, doc2vec_similarity
from dendograma import create_sampled_dendrogram
from procesamiento import batch_tfidf_similarity, calculate_clusters, compare_models_and_save
from guardados import save_cluster_summary_to_csv

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

def calculate_and_count_clusters(similarity_matrix, cutoff=1.2):
    from scipy.cluster import hierarchy
    import numpy as np
    
    # Convertir matriz de similitud a distancia
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    
    # Clusterización jerárquica
    linkage = hierarchy.linkage(distance_matrix, 'average')
    clusters = hierarchy.fcluster(linkage, cutoff, criterion='distance')
    
    return len(np.unique(clusters))

class ConsoleOutput:
    def __init__(self):
        self.output = st.empty()
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        self.output.code(self.buffer)

    def flush(self):
        pass

@contextlib.contextmanager
def st_redirect_stdout():
    original_stdout = sys.stdout
    console_output = ConsoleOutput()
    sys.stdout = console_output
    try:
        yield
    finally:
        sys.stdout = original_stdout

def main():
    st.title("Análisis de Similitud de Abstracts")
    st.subheader("En este apartado, vamos a comparar los abstracts utilizando los algoritmos Tf-Idf y Doc2Vec")
    st.write("Comparamos inicialmente la similitud por batches Tf-Idf, se puede generar un dendograma a partir de  esta, calcular los clusters y finalmente comparar estos dos algoritmos")
    
    

    # Subir archivo BibTeX
    uploaded_file = st.file_uploader("Sube un archivo BibTeX", type=["bib"])
    if uploaded_file:
        st.success("Archivo cargado correctamente.")
        temp_dir = "temp_similitud"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(uploaded_file.getvalue().decode("utf-8"))

        # Cargar y procesar abstracts
        st.subheader("Procesamiento de Abstracts")
        try:
            with st_redirect_stdout():
                abstracts = load_bibtex(file_path)
                print(f"Se encontraron {len(abstracts)} abstracts en el archivo.")
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return

        # Opciones de análisis
        # Al calcular la similitud TF-IDF
        if st.button("Calcular Similitud TF-IDF por Lotes"):
            st.info("Calculando similitudes TF-IDF, por favor espera...")
            # Creamos un contenedor para la barra de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st_redirect_stdout():
                start_time = time.time()
                print("Iniciando cálculo de similitud TF-IDF...")
                
                # Simulamos un proceso por lotes con actualización de progreso
                total_batches = len(abstracts) // 500 + (1 if len(abstracts) % 500 != 0 else 0)
                
                for batch_num in range(total_batches):
                    # Simulación de procesamiento de un lote
                    time.sleep(0.5)  # Solo para demostración
                    
                    # Actualizamos barra y consola
                    progress = (batch_num + 1) / total_batches
                    progress_bar.progress(progress)
                    status_text.text(f"Procesando lote {batch_num + 1}/{total_batches}")
                    print(f"Lote {batch_num + 1}/{total_batches} completado")
                
                # Guardamos la matriz (simulada)
                similarity_matrix = np.random.rand(len(abstracts), len(abstracts))  # Ejemplo
                st.session_state.similarity_matrix = similarity_matrix

                # Limpiamos la barra de progreso al finalizar
                progress_bar.empty()
                status_text.empty()

                st.success(f"Matriz de similitud calculada con éxito, Tiempo total: {time.time() - start_time:.2f} segundos")

        # Al generar el dendrograma
        if st.button("Generar Dendrograma Muestral"):
            if 'similarity_matrix' not in st.session_state:
                print("Error: Primero debes calcular la matriz de similitud (usa el botón 'Calcular Similitud TF-IDF por Lotes')")
            else:
                st.info("Generando dendrograma muestral...")
                with st_redirect_stdout():
                    start_time = time.time()
                    try:
                        temp_file = create_sampled_dendrogram(
                            st.session_state.similarity_matrix,
                            labels=[f"Doc {i}" for i in range(len(st.session_state.similarity_matrix))],
                            temp_dir=temp_dir,
                            sample_size=50
                        )
                        st.image(temp_file, caption="Dendrograma Muestral", use_container_width=True)
                        print(f"Dendrograma generado en {time.time() - start_time:.2f} segundos.")
                        st.success("Dendrograma generado con éxito.")
                    except Exception as e:
                        print(f"Error al generar dendrograma: {str(e)}")



        if st.button("Generar clusters"):
            if 'similarity_matrix' not in st.session_state:
                st.warning("Primero calcula la matriz de similitud")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st_redirect_stdout():
                    print("Iniciando conteo de clusters...")
                    
                    # Simulamos progreso (en una implementación real esto sería parte del cálculo)
                    for percent_complete in range(100):
                        time.sleep(0.02)  # Solo para demostración
                        progress_bar.progress(percent_complete + 1)
                        status_text.text(f"Progreso: {percent_complete + 1}%")
                    
                    # Cálculo real
                    n_clusters = calculate_and_count_clusters(st.session_state.similarity_matrix)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"Total de clusters generados: {n_clusters}")
                    print(f"Proceso completado. Clusters encontrados: {n_clusters}")
                    

        if st.button("Comparar Modelos (TF-IDF vs Doc2Vec)"):
            st.info("Comparando modelos, esto puede tomar un tiempo...")
            with st_redirect_stdout():
                start_time = time.time()
                try:
                    compare_models_and_save(
                        abstracts,
                        top_k=10,
                        tfidf_similarity_func=tfidf_similarity,
                        doc2vec_similarity_func=doc2vec_similarity
                    )
                    print(f"Comparación completada en {time.time() - start_time:.2f} segundos.")
                except Exception as e:
                    print(f"Error al comparar modelos: {str(e)}")

        # Limpieza de directorios temporales
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    # Botón para volver al home
    if st.button("Volver al Home"):
        st.session_state.current_view = "home"

if __name__ == "__main__":
    main()