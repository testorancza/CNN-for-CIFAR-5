from pathlib import Path
import streamlit as st
from Menu import menu

st.set_page_config(page_title='Strona startowa', page_icon='🏢')

menu()

st.image(image=str(Path(__file__).parent) + '/logo/logo.png', width=700)
st.subheader('Dedykowana aplikacja dla modelu konwolucyjnej sieci neuronowej')
st.divider()
st.markdown('''<h3 style='text-align: center;'>Opis funkcjonalności</h3>''', unsafe_allow_html=True)
st.write('')

titles = ['**Architektura i parametry modelu** - Opis architektury sieci wraz z poszczególnymi parametrami.',
          '**Podgląd zbioru danych** - Podgląd przykładowych obrazków każdej z 5 klas.',
          '**Wykresy procesu uczenia** - Interaktywne wykresy dokładności oraz funkcji straty dla zbioru treningowego i walidacyjnego.',
          '**Ocena modelu** - Statystyki modelu na zbiorze testowym.',
          '**Filtry** - Podgląd filtrów modelu przed oraz po treningu.',
          '**Mapy aktywacji klas** - Zobacz jakie obszary obrazu są istotne w procesie klasyfikacji.',
          '**Przewiduj** - Dokonaj predykcji obrazka ze zbioru testowego przy użyciu wytrenowanego modelu.',
          '**Przekaż obrazy** - Użyj modelu aby sklasyfikować własny obraz.',
          '**Oceń aplikację** - Oceń działanie aplikacji i zgłoś problemy.']

pages = ['Model_architecture', 'Dataset_preview', 'Learning_process_history', 'Model_evaluation', 'Filters',
         'Class_activation_mapping', 'Predict', 'Transfer_images', 'Rate']

for page in list(zip(titles,pages)):
    if st.button(page[0], use_container_width=True):
        st.switch_page(str(Path(__file__).parent) + '/pages/{}.py'.format(page[1]))
    st.write('')
