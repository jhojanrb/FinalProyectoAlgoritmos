import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

"""
Esta clase se encarga de crear un dendograma a partir de una matriz de similitud.
El objetivo es visualizar la estructura jerárquica de los documentos en función de su similitud.
Ademas que permite crear un dendograma utilizando una muestra de documentos, lo que es útil para visualizar grandes conjuntos de datos.
"""



# Funcion encargada de crear un dendograma a partir de una matriz de similitud
# y etiquetas de documentos. Esta funcion permite crear un dendograma
# utilizando una muestra de documentos, lo que es útil para visualizar grandes conjuntos de datos.

def create_sampled_dendrogram(similarity_matrix, labels, sample_size=100):
    """
    Crea un dendrograma usando una muestra de documentos.
    :param similarity_matrix: Matriz de similitud original.
    :param labels: Etiquetas de los documentos.
    :param sample_size: Tamaño de la muestra (número de documentos a considerar).
    """
    total_documents = len(labels)
    if sample_size > total_documents:
        sample_size = total_documents

    sampled_indices = np.random.choice(total_documents, size=sample_size, replace=False) # Seleccionar índices aleatorios
    sampled_similarity_matrix = similarity_matrix[np.ix_(sampled_indices, sampled_indices)] # Crear submatriz de similitud
    # Obtener etiquetas de la muestra
    sampled_labels = [labels[i] for i in sampled_indices]

    # Convertir similitud a distancia
    distance_matrix = 1 - sampled_similarity_matrix # Convertir a distancia
    # Forzar simetría
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix[distance_matrix < 0] = 0 # Asegurarse de que no haya valores negativos
    # Corregir la diagonal
    np.fill_diagonal(distance_matrix, 0)

    # Generar linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    # Crear gráfico
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=sampled_labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrograma de clustering jerárquico (muestra)")
    plt.xlabel("Documentos")
    plt.ylabel("Distancia")
    plt.savefig("C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/sampled_dendrogram100.png", dpi=300, bbox_inches='tight')

    # ------------------------------------------------ FUNCIONES ANTERIORES --------------------------------------------------#

    """

def create_dendrogram(similarity_matrix, labels=None):
    # Convertir similitud a distancia
    distance_matrix = 1 - similarity_matrix

    # Forzar simetría
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Asegurarse de que no haya valores negativos
    distance_matrix[distance_matrix < 0] = 0

    # Corregir la diagonal
    np.fill_diagonal(distance_matrix, 0)

    # Convertir a formato adecuado para linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    # Crear el dendrograma
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=None, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrograma de clustering jerárquico")
    plt.xlabel("Documentos")
    plt.ylabel("Distancia")
    plt.savefig("C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/dendrogram.png", dpi=300, bbox_inches='tight')



    """
