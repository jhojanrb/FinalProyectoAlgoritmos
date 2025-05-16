import streamlit as st
import os
import time
import re
from multiprocessing import Process, Queue
import sys

def scrape_acm(queue):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from playwright.sync_api import sync_playwright
    
    try:
        if not os.path.exists("Data"):
            os.makedirs("Data")

        with sync_playwright() as p:
            start_time = time.time()
            browser = p.chromium.launch(headless=True)  # Usar headless=True es más estable
            page = browser.new_page()

            # Paso 1: Acceder a la página principal
            page.goto("https://library.uniquindio.edu.co/databases", timeout=60000)
            page.wait_for_load_state("domcontentloaded")
            queue.put(("info", "Accedido a la página principal"))

            # Paso 2: Hacer clic en "Fac. Ingeniería"
            fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
            page.click(fac_ingenieria_selector)
            page.wait_for_load_state("domcontentloaded")
            queue.put(("info", "Accedido a la facultad de ingeniería"))

            # Paso 3: Hacer clic en "ACM Digital Library"
            elements = page.locator("//a[contains(@href, 'dl.acm.org')]//span[contains(text(), 'ACM Digital Library')]")
            count = elements.count()
            queue.put(("info", "Accediendo a la ACM Digital Library"))

            for i in range(count):
             if elements.nth(i).is_visible():
                elements.nth(i).click()
                page.wait_for_load_state("domcontentloaded")
                print(f"Se hizo clic en el elemento {i+1}")
                break
            else:
                print("No se encontró un elemento visible con el texto deseado.")    


            # Buscar artículos
            search_selector = "input[name='AllField']"
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")
            page.wait_for_selector(".search__item", timeout=60000)
            queue.put(("info", "Artículos encontrados"))

            # Cambiar a 50 artículos por página
            try:
                link_50_selector = "a[href*='pageSize=50']"
                page.wait_for_selector(link_50_selector, timeout=10000)
                page.click(link_50_selector)
                page.wait_for_load_state("domcontentloaded")
                print("Se seleccionó la opción de 50 artículos por página.")
            except Exception as e:
                print("No se encontró la opción de 50 artículos por página. Continuando con la configuración predeterminada.")

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "ACM_Interfaz.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                for page_num in range(1, 51):  # Iterar hasta la página 200
                    print(f"Procesando página {page_num}...")
                    queue.put(("info", f"Procesando página {page_num}..."))

                    # Revalidar que los resultados están disponibles
                    page.wait_for_selector(".search__item", timeout=60000)
                    results = page.query_selector_all(".search__item")

                    for i, result in enumerate(results):
                        try:
                            # Extraer información del artículo
                            title = result.query_selector(".hlFld-Title a").inner_text() if result.query_selector(".hlFld-Title a") else "Unknown"
                            link = result.query_selector(".hlFld-Title a").get_attribute("href") if result.query_selector(".hlFld-Title a") else "Unknown"
                            authors = result.query_selector(".rlist--inline").inner_text() if result.query_selector(".rlist--inline") else "Unknown"

                            year_element = result.query_selector(".bookPubDate")
                            year = re.search(r'\b\d{4}\b', year_element.inner_text()).group(0) if year_element and re.search(r'\b\d{4}\b', year_element.inner_text()) else "Unknown"
                            journal = result.query_selector(".issue-item__detail").inner_text() if result.query_selector(".issue-item__detail") else "Unknown"
                            abstract = result.query_selector(".issue-item__abstract").inner_text() if result.query_selector(".issue-item__abstract") else "Unknown"

                            # Escribir en formato BibTeX
                            file.write(f"@article{{ref{page_num}_{i},\n")
                            file.write(f"  title = {{{title}}},\n")
                            file.write(f"  author = {{{authors}}},\n")
                            file.write(f"  year = {{{year}}},\n")
                            file.write(f"  journal = {{{journal}}},\n")
                            file.write(f"  abstract = {{{abstract}}},\n")
                            file.write(f"  url = {{{'https://dl.acm.org' + link}}}\n")
                            file.write("}\n\n")
                        except Exception as e:
                            print(f"Error al procesar un resultado en la página {page_num}: {e}")
                            queue.put(("error", f"Error al procesar un resultado en la página {page_num}: {e}"))

                    # Avanzar a la siguiente página con reintentos
                    retries = 3
                    while retries > 0:
                        try:
                            next_button = page.query_selector(".pagination__btn--next")
                            if next_button:
                                next_button.click()
                                time.sleep(6)  # Esperar 3 segundos antes de cargar la siguiente página
                                page.wait_for_load_state("domcontentloaded", timeout=90000)  # Incrementar el tiempo de espera
                                break
                            else:
                                print("No se encontró el botón de siguiente. Finalizando.")
                                return
                        except Exception as e:
                            retries -= 1
                            print(f"Reintentando cargar la página {page_num + 1}. Intentos restantes: {retries}")
                            time.sleep(5)  # Pausa antes del siguiente intento
                    else:
                        print(f"Error al intentar cargar la página {page_num + 1}. Finalizando.")
                        queue.put(("error", f"Error al intentar cargar la página {page_num + 1}. Finalizando."))
                        break

            print(f"Los artículos se guardaron exitosamente en {filepath}")
            queue.put(("success", f"Los artículos se guardaron exitosamente en {filepath}"))
    except Exception as e:
            print(f"Error general: {e}")
            queue.put(("error", f"Error general: {e}"))
    finally:
            browser.close()
            end_time = time.time()
            print(f"Scraper para ACM finalizado en {end_time - start_time:.2f} segundos.\n")
            queue.put(("info", f"Scraper para ACM finalizado en {end_time - start_time:.2f} segundos.\n"))

def run_scraping_process():
    if not hasattr(st.session_state, 'scraping_queue'):
        st.session_state.scraping_queue = Queue()
        st.session_state.scraping_process = Process(
            target=scrape_acm,
            args=(st.session_state.scraping_queue,)
        )
        st.session_state.scraping_process.start()
        st.session_state.scraping_started = True

# Interfaz de Streamlit
st.title("ACM Digital Library Scraper")

if st.button("Iniciar Scraping"):
    run_scraping_process()
    st.success("Proceso de scraping iniciado en segundo plano")

# Mostrar resultados
if hasattr(st.session_state, 'scraping_started') and st.session_state.scraping_started:
    placeholder = st.empty()
    
    while hasattr(st.session_state, 'scraping_process') and st.session_state.scraping_process.is_alive():
        while not st.session_state.scraping_queue.empty():
            msg_type, message = st.session_state.scraping_queue.get()
            if msg_type == "info":
                placeholder.info(message)
            elif msg_type == "success":
                placeholder.success(message)
            elif msg_type == "error":
                placeholder.error(message)
        time.sleep(0.5)
    
    # Procesar últimos mensajes
    while not st.session_state.scraping_queue.empty():
        msg_type, message = st.session_state.scraping_queue.get()
        if msg_type == "info":
            st.info(message)
        elif msg_type == "success":
            st.success(message)
        elif msg_type == "error":
            st.error(message)
    
    st.session_state.scraping_process.join()
    del st.session_state.scraping_process
    del st.session_state.scraping_queue
    st.session_state.scraping_started = False