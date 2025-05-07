import os
import numpy as np
import pandas as pd


# Función para guardar el resumen de clusters en un archivo CSV
def save_cluster_summary_to_csv(clusters, abstracts, output_file='C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento5/cluster_summary.csv'):
    # Crear una lista para almacenar los datos
    cluster_data = []

    # Recorrer cada cluster y resumir los datos
    for cluster_id, abstract_indices in clusters.items():
        cluster_texts = [abstracts[i] for i in abstract_indices]
        cluster_summary = {
            "Cluster ID": cluster_id,
            "Número de Abstracts": len(abstract_indices),
            "Ejemplo de Abstract": cluster_texts[0] if cluster_texts else ""
        }
        cluster_data.append(cluster_summary)
    
    # Crear un DataFrame y guardarlo como CSV
    df = pd.DataFrame(cluster_data)
    df.to_csv(output_file, index=False)
    print(f"Resumen de clusters guardado en {output_file}")


    # ------------------------------ FUNCIONES ANTERIORES ------------------------------#

"""
def save_batch_results(matrix, batch_index, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"batch_{batch_index}.npy")
    np.save(file_path, matrix) 

def load_batch_results(output_dir):
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    matrices = [np.load(os.path.join(output_dir, f)) for f in files]
    return np.vstack(matrices)"""