import nltk
import time


from normalizacion5 import parse_large_bib, load_bibtex, preprocess
from algoritmos import tfidf_similarity, doc2vec_similarity
from dendograma import create_dendrogram, create_sampled_dendrogram
from procesamiento import batch_tfidf_similarity, calculate_clusters, compare_models_and_save
from guardados import save_batch_results, save_cluster_summary_to_csv


nltk.download('stopwords') # Descargar stopwords
# WordNet es una base de datos léxica del inglés que agrupa palabras en conjuntos de sinónimos (synsets), 
# proporcionando definiciones y ejemplos de uso.
nltk.download('wordnet') # Descargar wordnet


def main():
    # Cargar y procesar el archivo BibTeX
    file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    abstracts = load_bibtex(file_path)
    print(f"Cargando y procesando {len(abstracts)} abstracts...")
    #output_dir = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/similarity_batches"
    

    # Procesamiento por lotes batch para los 11k abstracts
    # Procesar similitud por lotes y guardar resultados
    print("Calculando similitud TF-IDF por lotes...")
    start_time = time.time() # Guardar tiempo de inicio
    similarity_matrix = batch_tfidf_similarity(abstracts, batch_size=500)
    #save_batch_results(similarity_matrix, 0, output_dir)
    end_time = time.time()
    print(f"Tiempo de cálculo por lotes: {end_time - start_time:.2f} segundos")

    # Generar dendrograma
    print("Generando dendrograma...")
    #dendograma completo
    #create_dendrogram(similarity_matrix, labels=[f"Doc {i}" for i in range(len(abstracts))])
    #dendograma con muestra
    start_time = time.time()
    create_sampled_dendrogram(similarity_matrix, labels=[f"Doc {i}" for i in range(len(similarity_matrix))], sample_size=100)
    end_time = time.time()
    print(f"Tiempo de generación de dendrograma: {end_time - start_time:.2f} segundos")
    # batch_process(abstracts)

    # Calcular clusters en toda la matriz
    print("\nCalculando clusters...")
    cutoff_distance = 0.8  # Parámetro de corte para definir los clusters
    clusters, _ = calculate_clusters(similarity_matrix, cutoff_distance)

    # Guardar resumen de clusters en CSV
    print("Guardando resumen de clusters")
    save_cluster_summary_to_csv(clusters, abstracts)
    
    # Comparar resultados para un abstract específico
    print("\nComparando modelos...")
    #tiempo de comparacion
    start_time = time.time()
    #compare_models(abstracts, doc_index=0, top_k=10)
    compare_models_and_save(abstracts, top_k=10, tfidf_similarity_func=tfidf_similarity, doc2vec_similarity_func=doc2vec_similarity)
    end_time = time.time()
    print(f"\nTiempo de comparación: {end_time - start_time:.2f} segundos")
    

if __name__ == "__main__":
    main()


    # ------------------------------------------------ FUNCIONES ANTERIORES --------------------------------------------------#

    # Opción 1: Procesamiento completo (requiere recursos)
    """
    start_time = time.time()
    tfidf_sim = tfidf_similarity(abstracts)
    print("\nSimilitud TF-IDF calculada.")
    end_time = time.time()
    print(f"Tiempo de cálculo TF-IDF: {end_time - start_time:.2f} segundos")
    start_time = time.time()
    doc2vec_sim = doc2vec_similarity(abstracts)
    print("\nSimilitud Doc2Vec calculada.")
    end_time = time.time()
    print(f"Tiempo de cálculo Doc2Vec: {end_time - start_time:.2f} segundos")"""