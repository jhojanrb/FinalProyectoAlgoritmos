import streamlit as st

def home_view():
    st.title("Proyecto Final Algoritmos 2025-1")
    st.subheader("Bienvenido a la interfaz de usuario del proyecto final de Algoritmos 2025-1")
    st.write("Este proyecto tiene como objetivo la realizacion de un analisis bibliométrico y la realizacion de ciertos requerimeintos brindados por el profesor Sergio Augusto Cardona Torres.")
    st.write("Realizado por:")
    st.write("- Erica Paola Rueda")
    st.write("- Jhojan Ramirez Botache")
    st.write("- Sebastian Bohorquez Coy")

    
    st.write("Selecciona una funcionalidad para continuar:")
    if st.button("Unificar Archivos BibTeX"):
        st.session_state.current_view = "unify"
    if st.button("Generar Estadísticas"):
        st.session_state.current_view = "estadisticas"
    if st.button("Categorias y sus variables"):
        st.session_state.current_view = "categorias"
    if st.button("Funcionalidad 4"):
        st.session_state.current_view = "function4"
