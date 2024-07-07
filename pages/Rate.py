from pathlib import Path
import streamlit as st
from Menu import menu
import os
import streamlit_antd_components as sac
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Oceń aplikację', page_icon='🗳️', layout='wide')

menu()

st.title('Oceń aplikację')
st.markdown('Opinia każdego użytkownika jest ważna. Oceń działanie aplikacji oraz zgłoś ewentualne błędy.')

cols = st.columns(2, gap='large')

with cols[0]:
    st.header('Oceń aplikację')
    rate = sac.rate(label='Dokonaj oceny', value=0.0, align='start', size='lg', half=True)
    comment = st.text_area('Twój komentarz', height=300)
    if ui.button("Prześlij opinię", key="clk_btn"):
        st.toast('Dziękuję za opinię.')
        with open(str(Path(__file__).parents[1]) + '/pages/Rate/Reviews.txt', 'a') as file:
            if rate:
                file.write('Ocena: {}.\n'.format(rate))
            if comment:
                file.write('Komentarz: {}\n'.format(comment))
            file.write('------------------\n')

with cols[1]:
    st.header('Zgłoś błąd')
    option = st.selectbox('Wybierz miejsce wystąpienia błędu',
                          options=['Architektura i parametry sieci', 'Podgląd zbioru danych', 'Wykresy procesu uczenia',
                                   'Ocena modelu', 'Filtry', 'Mapy aktywacji klas', 'Przewiduj', 'Przekaż obrazy',
                                   'Oceń aplikację', 'Inne'],
                          index=None, placeholder='...')
    report = st.text_area('Opisz problem', height=300)
    upload_file = st.file_uploader("Przekaż obraz")
    if ui.button("Zgłoś błąd", key="clk_btn2"):
        st.toast('Twoje zgłoszenie zostało przesłane. Dziękuję.')
        with open(str(Path(__file__).parents[1]) +'/pages/Rate/Reports.txt', 'a') as file2:
            if rate:
                file2.write('Miejsca błędu: {}.\n'.format(option))
            if comment:
                file2.write('Opis: {}\n'.format(report))
            file2.write('------------------\n')
        if upload_file:
            with open(os.path.join(str(Path(__file__).parents[1]) + '/pages/Rate/', upload_file.name), "wb") as file3:
                file3.write(upload_file.getbuffer())




