import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from normalizacion5 import preprocess


def tfidf_similarity(abstracts):
    processed_abstracts = [' '.join(preprocess(ab)) for ab in abstracts]
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Limitar el vocabulario a las 5K palabras más frecuentes
        ngram_range=(1, 3),      # Incluir unigramas, bigramas y trigramas
        stop_words='english'     # Eliminar stopwords (redundante con preprocess, pero útil)
    )
    tfidf_matrix = vectorizer.fit_transform(processed_abstracts)
    return cosine_similarity(tfidf_matrix)



def doc2vec_similarity(abstracts, save_model=True):
    tagged_data = [TaggedDocument(preprocess(ab), [str(i)]) for i, ab in enumerate(abstracts)]
    
    model = Doc2Vec(
        vector_size=100,
        min_count=2,
        epochs=20,               # Reducir epochs para velocidad (20 suele ser suficiente)
        dm=1,
        workers=4                # Paralelizar en 4 núcleos
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    if save_model:
        model.save("C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/doc2vec_model.model")  # Guardar modelo para reutilizar
    
    # Calcular similitud solo para los top N más similares (ej: top 100)
    top_n = 1000
    similarity_matrix = []
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=top_n)
        similarity_matrix.append(sims)  # Guardar (índice, similitud) en lugar de matriz completa
    
    return similarity_matrix