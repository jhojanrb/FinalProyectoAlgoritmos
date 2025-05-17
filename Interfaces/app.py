import streamlit as st

# Importamos las vistas para llamar sus funciones
from home import home_view
from unificacion import unification_view
from estadisticas_view import estadisticas_view
from categorias import categorias_view
from similitud import main 

# Configuración inicial de la vista
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# Navegación entre vistas
if st.session_state.current_view == "home":
    home_view()
elif st.session_state.current_view == "unify":
    unification_view()
elif st.session_state.current_view == "estadisticas":
    estadisticas_view()
elif st.session_state.current_view == "categorias":
    categorias_view()
elif st.session_state.current_view == "similitud":
    main()
