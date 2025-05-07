import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from normalizacion5 import preprocess

"""
Esta clase se encarga de calcular la similitud entre abstracts utilizando diferentes métodos.
El objetivo es facilitar el análisis de similitud entre documentos y la creación de clusters.
Ademas que permite calcular la similitud entre abstracts utilizando el modelo TF-IDF y el modelo Doc2Vec."""


# Función para calcular la similitud TF-IDF
# Esta función calcula la similitud entre abstracts utilizando el modelo TF-IDF.
def tfidf_similarity(abstracts):
    processed_abstracts = [' '.join(preprocess(ab)) for ab in abstracts] # Preprocesar abstracts
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Limitar el vocabulario a las 5K palabras más frecuentes
        ngram_range=(1, 3),      # Incluir unigramas, bigramas y trigramas
        stop_words='english'     # Eliminar stopwords (redundante con preprocess, pero útil)
    )
    tfidf_matrix = vectorizer.fit_transform(processed_abstracts) # Calcular matriz TF-IDF
    return cosine_similarity(tfidf_matrix) 

# Función para calcular la similitud Doc2Vec
# Esta función utiliza el modelo Doc2Vec para calcular la similitud entre abstracts.
def doc2vec_similarity(abstracts, save_model=True):
    tagged_data = [TaggedDocument(preprocess(ab), [str(i)]) for i, ab in enumerate(abstracts)] # Etiquetar documentos
    
    model = Doc2Vec(
        vector_size=100,
        min_count=2,
        epochs=20,               # Reducir epochs para velocidad (20 suele ser suficiente)
        dm=1,
        workers=4                # Paralelizar en 4 núcleos
    )
    model.build_vocab(tagged_data) # Construir vocabulario
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs) # Entrenar modelo
    
    if save_model:
        model.save("C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/doc2vec_model.model")  # Guardar modelo para reutilizar
    
    # Calcular similitud solo para los top N más similares (ej: top 1000)
    top_n = 1000
    similarity_matrix = []
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=top_n)
        similarity_matrix.append(sims)  # Guardar (índice, similitud) en lugar de matriz completa
    
    return similarity_matrix