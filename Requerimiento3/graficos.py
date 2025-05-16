import matplotlib.pyplot as plt
import os
import networkx as nx
from wordcloud import WordCloud
import matplotlib.patheffects as path_effects
import pandas as pd
from collections import Counter
from itertools import combinations
import matplotlib.patches as mpatches
import tempfile


# Ruta donde se guardarán los gráficos
ruta_graficos = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento3"


# Funcion para graficar el conteo de palabras clave
# Esta función toma un diccionario de conteos de palabras clave y genera un gráfico de barras
# mostrando las 20 palabras clave más frecuentes.
def plot_bar_chart(keyword_counts, temp_dir):
    
    top_20 = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Separar claves y valores
    keywords = [item[0] for item in top_20]
    counts = [item[1] for item in top_20]

    plt.figure(figsize=(12, 6))
    plt.barh(keywords, counts, color="skyblue")

    #Agregar etiquetas al final de la barra
    for i, count in enumerate(counts):
        plt.text(count + 0.5, i, str(count), va='center', fontsize=10)

    plt.xlabel("Frecuencia")
    plt.title("Top 20 - Frecuencia de Palabras Clave")
    plt.tight_layout()
    # Guardar la figura en el directorio temporal
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir).name
    plt.savefig(temp_file_path)
    plt.close()

    return temp_file_path  # Retornar la ruta del archivo guardado

# Función para generar una nube de palabras a partir de los conteos de palabras clave
# Esta función toma un diccionario de conteos de palabras clave y genera una nube de palabras
def generate_wordcloud(keyword_counts, temp_dir):
        wordcloud = WordCloud(
        width=1600,              # Mayor resolución horizontal
        height=800,              # Mayor resolución vertical
        background_color="white",
        colormap="tab10",        # Mejores colores (puedes probar: "viridis", "plasma", "Set2", etc.)
        prefer_horizontal=0.9,   # Preferir palabras en horizontal
        max_words=200,           # Aumenta el número de palabras visibles
        contour_color='black',   # Borde negro (opcional)
        contour_width=0.5        # Grosor del borde
    ).generate_from_frequencies(keyword_counts)

        plt.figure(figsize=(16, 8))  # Lienzo más grande
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)

        temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir).name
        plt.savefig(temp_file_path)  # Alta resolución
        plt.close()

        return temp_file_path
        
# Función para leer palabras clave y categorías desde un archivo Excel 
def cargarPalabras_excel(file_path):
    df = pd.read_excel(file_path)
    keywords_by_category = df.groupby("Categoría")["Término"].apply(list).to_dict()
    return keywords_by_category

# Funcion encargada de realizar el grafico de co-occurence network
def plot_cooccurrence_network(keywords_by_category, temp_dir, min_cooccurrence=1):
    """Generar gráfico de red de co-ocurrencia con categorías."""

    G = nx.Graph() # Crear un grafo vacío

    # Contar co-ocurrencias
    co_occurrence = Counter()
    # Iterar sobre cada categoría y sus palabras clave
    for category, keywords in keywords_by_category.items():
        for kw1, kw2 in combinations(keywords, 2):
            co_occurrence[(kw1, kw2)] += 1

    # Paleta de colores más variada
    category_list = list(keywords_by_category.keys())
    color_map = {cat: plt.colormaps["Set3"].resampled(len(category_list))(i) for i, cat in enumerate(category_list)}

    # Agregar nodos
    for category, keywords in keywords_by_category.items():
        for keyword in keywords:
            if not G.has_node(keyword):
                G.add_node(keyword, category=category)

    # Agregar aristas
    for (kw1, kw2), weight in co_occurrence.items():
        if weight >= min_cooccurrence:
            G.add_edge(kw1, kw2, weight=weight)

    # Calcular tamaños por grado ponderado
    degrees = dict(G.degree(weight="weight"))
    max_degree = max(degrees.values()) if degrees else 1
    # Normalizar tamaños de nodos
    sizes = {node: 100 + 1000 * (deg / max_degree) for node, deg in degrees.items()}

    # Layout
    pos = nx.spring_layout(G, k=2.2, iterations=100, seed=42)

    plt.figure(figsize=(24, 20))

    # Dibujar nodos
    for category, color in color_map.items():
        nodes = [n for n in G.nodes if G.nodes[n]["category"] == category]
        node_sizes = [sizes[n] for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_sizes, node_color=[color]*len(nodes), alpha=0.9)

    # Dibujar aristas
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    weights_scaled = [0.5 + w*0.3 for w in weights]
    nx.draw_networkx_edges(G, pos, width=weights_scaled, edge_color="lightgray", alpha=0.6)

    # Dibujar etiquetas
    labels = {n: n for n in G.nodes()}
    text = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    for t in text.values():
        t.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    # Crear parches de leyenda
    legend_patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
    plt.legend(handles=legend_patches, title="Categorías", loc="lower left", fontsize=10, title_fontsize=11, frameon=True)

    plt.title("Red de Co-Ocurrencia de Palabras Clave por Categorías", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    # Guardar la figura en el directorio temporal
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir).name
    plt.savefig(temp_file_path)
    plt.close()

    return temp_file_path  # Retornar la ruta del archivo guardado


