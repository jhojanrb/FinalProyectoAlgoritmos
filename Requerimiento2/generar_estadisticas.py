import pandas as pd
import os
import sys

"""
Esta clase se encarga de generar las estadisticas necesarias a partir de un DataFrame
como lo es obtener el top 15 de autores, la distribucion por tipo, la cantidad por tipo y año,
el top de journals por tipo y el top de publishers por año.
Además, guarda las estadisticas en un archivo Excel.
"""

# Agregar la carpeta raíz al sys.path
# para poder importar módulos desde allí
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Requerimiento2.limpieza_normalizacion import normalize_product_type, clean_journal_name

# Funcion encargada de generar estadísticas a partir de un DataFrame
def generate_statistics(df):
    """Genera estadísticas completas"""
    stats = {}
    
    # Usar el tipo normalizado para las estadísticas
    if 'tipo_normalizado' not in df.columns:
        df['tipo_normalizado'] = df['tipo'].apply(normalize_product_type)
    
    # 1. Top autores
    # Filtramos los autores y los ordenamos por cantidad de publicaciones (15 autores)
    stats['top_15_autores'] = (
        df.explode('author')
        .dropna(subset=['author'])
        .groupby('author')
        .size()
        .sort_values(ascending=False)
        .head(15)
    )

    # 2. Distribución por tipo
    # Filtramos los tipos y los ordenamos por cantidad de publicaciones
    stats['tipos_existentes'] = df['tipo_normalizado'].value_counts() # devuelve la cantidad de tipos existentes
    
    # 3. Cantidad por tipo y año
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')[0] # Extraer solo el año
        stats['añoPublicacion_Portipo'] = (
            df.groupby(['tipo_normalizado', 'year']) # Agrupamos por tipo y año
            .size()
            .unstack(fill_value=0)
        )
    
    # 4. Top journals por tipo
    # Filtramos los journals y los agrupamos por tipo y journal
    if 'journal' in df.columns:
        df['journal_clean'] = df['journal'].apply(clean_journal_name) # Normalizamos los nombres de los journals
    
    # Obtenemos los conteos por tipo y journal
    journal_counts = (
        df.groupby(['tipo_normalizado', 'journal_clean'])
        .size()
        .unstack(level=0, fill_value=0)
    )
    
    # Calculamos el total por journal (suma horizontal)
    journal_counts['Total'] = journal_counts.sum(axis=1)
    
    # Ordenamos por el total y seleccionamos los top 15
    top_journals = (
        journal_counts
        .sort_values('Total', ascending=False)
        .head(15)
    )
    
    # Reorganizamos columnas para que Total esté primero
    cols = ['Total'] + [col for col in top_journals.columns if col != 'Total']
    stats['top15_journals'] = top_journals[cols]
    
    # 5. Top publishers por año
    # Filtramos los publishers, los normalizamos y obtenemos el top 15
    if 'publisher' in df.columns:
        df['publisher_clean'] = df['publisher'].apply(clean_journal_name)
        
        # Obtenemos los conteos por tipo y publisher
    publisher_counts = (
        df.groupby(['tipo_normalizado', 'publisher_clean'])
        .size()
        .unstack(level=0, fill_value=0)
    )
    
    # Calculamos el total por publisher (suma horizontal)
    publisher_counts['Total'] = publisher_counts.sum(axis=1)
    
    # Ordenamos por el total y seleccionamos los top 15
    top_publishers = (
        publisher_counts
        .sort_values('Total', ascending=False)
        .head(15)
    )
    
    # Reorganizamos columnas para que Total esté primero
    cols = ['Total'] + [col for col in top_publishers.columns if col != 'Total']
    stats['top15_publishers'] = top_publishers[cols]
        
    return stats

# Funcion encargada de guardar las estadísticas en un archivo Excel
def save_statistics(stats, output_path):
    """Guarda estadísticas en archivo Excel"""
    with pd.ExcelWriter(output_path) as writer:
        for name, data in stats.items():
            data.to_excel(writer, sheet_name=name[:31])
