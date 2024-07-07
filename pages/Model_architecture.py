from pathlib import Path
import streamlit as st
from Menu import menu
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Architektura i parametry modelu', page_icon='🏗️', layout='wide')

menu()

st.title('Architektura i parametry modelu')
st.write('')

cols = st.columns(9)
with cols[1]:
    st.image(str(Path(__file__).parents[1]) + '/pages/Model_architecture/Architecture.jpg', width=950)

st.divider()
cols2 = st.columns(6)

with cols2[0]:
    ui.metric_card(title='Zbiór danych', content='Obrazy RGB 5 klas w rozmiarze 32x32 pikseli', description='zbiór treningowy: 2000, zbiór walidacyjny: 500, zbiór testowy: 500')
with cols2[1]:
    ui.metric_card(title='Warstwa konwolucyjna', content='16 filtrów w rozmiarze 3x3x3 pikseli', description='Stride: 1, Padding: 1')
with cols2[2]:
    ui.metric_card(title='Warstwa łącząca', content='MaxPooling', description='rozmiar 2x2 pikseli, stride: 2')
with cols2[3]:
    ui.metric_card(title='Optymalizator', content='Adaptive Moment Estimation (ADAM)', description='współczynnik uczenia: 0.001')
with cols2[4]:
    ui.metric_card(title='Funkcje straty i aktywacji', content='Entropia krzyżowa Softmax')
with cols2[5]:
    ui.metric_card(title='Proces uczenia', content='Liczba epok: 30 wielkość paczki: 32', description='wielkość wag: 0.001, regularyzacja L2: 0.001')