import streamlit as st

def home_view():
    st.title("Home - Proyecto Final Algoritmos 2025-1")
    
    st.write("Selecciona una funcionalidad para continuar:")
    if st.button("Unificar Archivos BibTeX"):
        st.session_state.current_view = "unify"
    if st.button("Generar Estadísticas"):
        st.session_state.current_view = "estadisticas"
    if st.button("Funcionalidad 3"):
        st.session_state.current_view = "function3"
    if st.button("Funcionalidad 4"):
        st.session_state.current_view = "function4"
