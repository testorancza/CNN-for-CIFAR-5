import streamlit as st

def menu():
    st.sidebar.page_link('App.py', label='Strona startowa', icon='🏢')
    st.sidebar.page_link('pages/Model_architecture.py', label='Architekura i parametry modelu', icon='🏗️')
    st.sidebar.page_link('pages/Dataset_preview.py', label='Podgląd zbioru danych', icon='🗃️')
    st.sidebar.page_link('pages/Learning_process_history.py', label='Wykresy procesu uczenia', icon='📊')
    st.sidebar.page_link('pages/Model_evaluation.py', label='Ocena modelu', icon='📐')
    st.sidebar.page_link('pages/Filters.py', label='Filtry', icon='🎞️')
    st.sidebar.page_link('pages/Class_activation_mapping.py', label='Mapy aktywacji klas', icon='🗺️')
    st.sidebar.page_link('pages/Predict.py', label='Przewiduj', icon='💭')
    st.sidebar.page_link('pages/Transfer_images.py', label='Przekaż obrazy', icon='📥')
    st.sidebar.page_link('pages/Rate.py', label='Oceń aplikację', icon='🗳️')