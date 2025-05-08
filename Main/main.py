import sys
import os
import time  # Importar módulo para medir el tiempo

# Agregar la carpeta raíz al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Scrape.Unificar import unify_results_from_files



if __name__ == "__main__":
    try:
        
        # Unificación de resultados
        print("Unificando resultados de todos los scrapers...")
        start_time = time.time()
        unify_results_from_files("Data/resultados_ieee.bib", 
                                 "Data/resultados_springer_open.bib", 
                                 "Data/resultados_ACM.bib")
        end_time = time.time()
        print(f"Unificación finalizada en {end_time - start_time:.2f} segundos.")
        print("Datos unificados y duplicados almacenados correctamente.")

    except Exception as e:
        print(f"Se produjo un error durante la ejecución: {e}")
